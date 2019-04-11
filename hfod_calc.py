try:
    from testkit import sjutils, logger, settings
except ImportError:
    pass

# def TEST_x():
#     logger.debug(preview_expression())

from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import string
import random
import yaml


def sid(things, id=None, for_query=True):
    if things not in ('Companies', 'LineItems', 'Reports'):
        raise Exception('invalid things: {}'.format(things))
    sid = datastore.sid(things, id) if id else datastore.sid(things)
    if for_query:
        sid = 'sid:{}'.format(sid)
    return sid


def process():
    # for now this is just based on daily run;
    # later should have mid-day modes too (based on things that changed throughout the day)

    # company stuff
    companies = list(sjclient.scroll(r'sid:hfod\companies peer_group=n', fields=['*']))
    last_finalized_company_reports = defaultdict(int)
    for s in sjclient.scroll(r'sid:hfod\reports period=yr', fields=['company', 'year'], scroll_batch_size=1000):
        last_finalized_company_reports[s['fields']['company']] = max(last_finalized_company_reports[s['fields']['company']], int(s['fields']['year']))
    last_finalized_company_reports_w_is_ni_reported_num = defaultdict(int)
    for s in sjclient.scroll(r'sid:hfod\reports period=yr v_obj.l=is_ni_reported_num', fields=['company', 'year'], scroll_batch_size=1000):
        last_finalized_company_reports_w_is_ni_reported_num[s['fields']['company']] = max(last_finalized_company_reports_w_is_ni_reported_num[s['fields']['company']], int(s['fields']['year']))
    last_finalized_company_reports_w_ci_sum_num = defaultdict(int)
    for s in sjclient.scroll(r'sid:hfod\reports period=yr v_obj.l=ci_sum_num', fields=['company', 'year'], scroll_batch_size=1000):
        last_finalized_company_reports_w_ci_sum_num[s['fields']['company']] = max(last_finalized_company_reports_w_ci_sum_num[s['fields']['company']], int(s['fields']['year']))
    for c in companies:
        cid = c['fields']['identifier']
        if cid not in last_finalized_company_reports:
            continue
        report_sid = 'HFOD\\Reports\\{}\\{}\\yr\\usd'.format(cid, last_finalized_company_reports[cid])
        v_obj = {v['l']: v['v_num'] for v in sjclient.get_series('sid={}'.format(report_sid), fields=['v_obj'])['fields']['v_obj'] if v['r'] == 'WWF'}
        flds = {
         'd_obj.last_gi_fiscalpe_num': int(v_obj.get('gi_fiscalpe_num')) if v_obj.get('gi_fiscalpe_num') else None,
         'd_obj.last_finalized_year': last_finalized_company_reports[cid] or None,
         'd_obj.last_year_w_is_ni_reported': last_finalized_company_reports_w_is_ni_reported_num[cid] or None,
         'd_obj.last_year_w_ci_sum': last_finalized_company_reports_w_ci_sum_num[cid] or None,
        }
        datastore.put_fields('HFOD\\Companies\\{}'.format(cid), flds)

    return
    # lookups
    company_lu = {c['fields']['identifier']: c['fields'] for c in sjclient.scroll(sid('Companies'), fields=['*'])}
    # companny rendered info
    for c_id, c in company_lu.items():
        w = False
        d_obj = {}
        if 'parent_tree' in c:
            named_parent_tree = u'\\'.join([company_lu.get(p_id, {'description': p_id}).get('description', p_id) for p_id in c['parent_tree'].split('\\')])
            d_obj['named_parent_tree'] = named_parent_tree
        if d_obj != c.get('d_obj'):
            datastore.put_field(sid('Companies', c_id, for_query=False), 'd_obj', d_obj)
        c['d_obj'] = d_obj
    # reports rendered info
    for r in sjclient.scroll(sid('Reports'), fields=['company', 'd_obj'], scroll_batch_size=10):  # TODO: shouldn't do  currency here! all we need is fields=['company','d_obj'] for now
        d_obj = {}
        if r['fields']['company'] in company_lu:  # there's a company lookup!
            c = company_lu[r['fields']['company']]
            d_obj['company_name'] = c.get('description')
            d_obj['named_parent_tree'] = c.get('d_obj', {}).get('named_parent_tree')
            d_obj['company_id'] = c.get('company_id')
            #cvs = [v['v_num'] for v in r['fields']['v_obj'] if v['l']=='gi_currencycode']
            #d_obj['currency'] = cvs[0] if len(cvs) else None
        if d_obj != r['fields'].get('d_obj'):
            datastore.put_field(r['series_id'], 'd_obj', d_obj)
    # line items rendered info (category sort mostly?)


class NotMeaningful(Exception):
    pass

### Callables
@sjutils.callable
def get_full_report():
    """
    Gets report fields from Shooju and performs several necessary calcs on input HFOD\Report data;
    removes dynamic and non-leaf fields, applies special data structure, and adds geographic rollups.
    """
    rc = ReportCalc(settings['company'], settings['year'], settings['period'], settings.get('currency', 'usd'))
    rc.calc_report()
    return rc.report_for_ui


@sjutils.callable
def save_line_item():
    """Saves a line item."""
    # first check if line item exists; if we're overwriting? do we have permissions to do this?
    for do_not_save_field in ['company', 'periodicity', 'region', 'widgetselected']:
        settings.pop(do_not_save_field, None)
    with sjclient.register_job('save line item') as job:
        job.put_fields('HFOD\\LineItems\\{}'.format(settings['fields']['named_id']), settings['fields'])
        if settings['delete_line_item'] and settings['delete_line_item'].lower() != settings['fields']['named_id'].lower():
            job.delete('HFOD\\LineItems\\{}'.format(settings['delete_line_item']))  # TODO: - Dangerous! check for dependencies first
    return {'saved': True}


@sjutils.callable
def reorder_line_items():
    return {'got_settings': settings}


@sjutils.callable
def get_geo_tree():
    """Returns family trees of all root nodes as dict of {name: tree}."""
    tree = {}
    for region, params in GEO.items():
        if params.get('parent') is None:
            tree[region.lower()] = get_family_tree(region)
    return {"tree": tree, "name_lookup": {r.lower(): d['name'] for r, d in GEO.items()}}


@sjutils.callable
def preview_expression():
    # periodicty options: Yr / Q# / H1 / M9
    # Store values from settings
    dynamic_obj = settings.get('dynamic_obj', {})
    company = settings['company']
    periodicity = settings['periodicity']
    # Store scroll query
    q = 'sid:HFOD\\Reports company="{}"  period={} currency={}'.format(company, periodicity, settings.get('currency', 'usd'))
    logger.debug('using query: {}'.format(q))
    results = []
    # For each report available for the company, period and currency specified...
    for s in sjclient.scroll(q, fields=['year', 'period'], sort='year desc,period desc'):
        # Get report and perform geo rollups
        logger.debug('doing {}/{}/{}'.format(company, s['fields']['year'], s['fields']['period']))
        dynamic_obj_periodized = periodize_dynamic_obj(dynamic_obj, s['fields']['period'])
        logger.debug('{} -> {}'.format(dynamic_obj, dynamic_obj_periodized))
        if dynamic_obj_periodized is None:
            continue
        rc = ReportCalc(company, s['fields']['year'], s['fields']['period'], settings.get('currency', 'usd'))
        by_region = rc.simulate_dynamic_obj_for_ui(dynamic_obj_periodized)

        results.append({
            'year': s['fields']['year'],
            'period': s['fields']['period'],
            'by_region': {k: v for k, v in by_region.items() if v['v'] is not None},
        })
    return results


@sjutils.callable
def save_adhoc_report_config():
    """Saves a adhoc report config."""
    # first check if line item exists; if we're overwriting? do we have permissions to do this?
    req_fields = ('companies', 'lineitems', 'regions', 'display_rows', 'display_columns', 'description')
    for req_field in req_fields:
        if req_field not in settings['adhoc_report_config']:
            raise Exception('one of required fields is missing: {}'.format(', '.join(req_fields)))
        if 'periods_obj' not in settings['adhoc_report_config']:
            if 'periods_num' not in settings['adhoc_report_config'] or 'periods_type' not in settings['adhoc_report_config']:
                raise Exception('either periods_obj or periods_num+periods_type are required')
    if 'adhoc_report_config_id' not in settings:
        adhoc_report_config_id = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(8))
    else:
        adhoc_report_config_id = settings['adhoc_report_config_id']
    with sjclient.register_job('save adhoc report conf') as job:
        settings['adhoc_report_config']['last_saved_by'] = caller_user
        job.put_fields('HFOD\\AdHocReportConfigs\\{}'.format(adhoc_report_config_id), settings['adhoc_report_config'])
    return {'adhoc_report_config_id': adhoc_report_config_id, 'saved': True}


@sjutils.callable
def delete_adhoc_report_config():
    """Deletes a adhoc report config."""
    adhoc_report_config_id = settings['adhoc_report_config_id']
    with sjclient.register_job('del adhoc report conf') as job:
        job.delete('HFOD\\AdHocReportConfigs\\{}'.format(adhoc_report_config_id))
    return {'deleted': True}


def trigger():
    logger.debug(xpr_load_adhoc_report('Mb2reVNb'))


#@sjutils.expressionable()
def xpr_load_adhoc_report(report_id):
    a = sjclient.get_series('HFOD\\AdHocReportConfigs\\{}'.format(report_id), fields=['*'])
    if a is None:
        raise ImporterException('adhoc reprot config w/ id {} not found'.format(settings['adhoc_report_config_id']))
    adhoc_report_config = a['fields']
    cells = []
    for c in adhoc_report_config['companies']:
        for p in adhoc_report_config['periods_obj']:
            period_to_month = {
                'q1': 1,
                'q2': 4,
                'q3': 7,
                'q4': 10,
            }
            dt = date(int(p['year']), period_to_month.get(p['period'], 1), 1)
            rc = ReportCalc(c, p['year'], p['period'], 'usd')
            rc.calc_report()
            dynamic_line_items = {l['field'] for l in get_dynamic_objs(p['period'])}
            for r in adhoc_report_config['regions']:
                for l in adhoc_report_config['lineitems']:
                    val = rc.report.get(r2f(r), {}).get(l)
                    if val is not None:
                        cell = {
                            'c': c,
                            'dt': dt,
                            'r': r,
                            'l': l,
                            'v': val.value
                        }
                        cells.append(cell)
    series = defaultdict(dict)
    for c in cells:
        sid = '{}\\{}\\{}'.format(c['c'], c['r'], c['l'])
        series[sid][c['dt']] = c['v']
        sjutils.xpr_fields(sid, {
            'country': c['c'],
            'region': c['r'],
            'lineitem': c['l']
        })
    import pandas as pd
    for sid, pts in series.items():
        pts = {datetime(*d.timetuple()[:3]): v for d, v in pts.iteritems()}
        series[sid] = pd.Series(pts)
    return pd.DataFrame(series)


@sjutils.callable
def load_adhoc_report():
    """Loads a adhoc report."""
    if 'adhoc_report_config_id' in settings:
        a = sjclient.get_series('HFOD\\AdHocReportConfigs\\{}'.format(settings['adhoc_report_config_id']), fields=['*'])
        if a is None:
            return {'error': 'adhoc reprot config w/ id {} not found'.format(settings['adhoc_report_config_id'])}
        adhoc_report_config = a['fields']
    elif 'adhoc_report_config' not in settings:
        raise Exception('ether adhoc_report_config_id or adhoc_report_config must be passed')
    else:
        adhoc_report_config = settings['adhoc_report_config']
    cells = []
    for c in adhoc_report_config['companies']:
        for p in adhoc_report_config['periods_obj']:
            period_string = '{}: {}'.format(p['year'], p['period'])
            overrides = []
            for cell in settings.get('edited_cells', []):
                if cell['p'] == period_string and cell['c'] == c:
                    overrides.append({'l': cell['l'], 'r': cell['r'], 'v': cell['v']})
            rc = ReportCalc(c, p['year'], p['period'], 'usd', overrides=overrides)
            rc.calc_report()
            dynamic_line_items = {l['field'] for l in get_dynamic_objs(p['period'])}
            for r in adhoc_report_config['regions']:
                for l in adhoc_report_config['lineitems']:
                    val = rc.report.get(r2f(r), {}).get(l)
                    if val is not None:
                        cell = {
                            'c': c,
                            'p': period_string,
                            'r': r,
                            'l': l,
                            'v': val.value,
                            'e': val.input
                        }
                        cells.append(cell)
    return {'cells': cells}


def test_load_adhoc_report():
    with updated_settings({'adhoc_report_config': {
        'companies': ['test_1', 'test_2'],
        'lineitems': ['revenue_num', 'cost_num', 'profit_num'],
        'regions': ['US', 'WWR'],
        'display_rows': ['companies'],
        'display_columns': ['regions'],
        'periods_obj': [
            {'year': '2017', 'period': 'yr'},
            {'year': '2017', 'period': 'q1'},
        ]
    }}):
        cells = load_adhoc_report()['cells']
        assert len(cells) == 18
        profit = [c for c in cells if c['l'] == 'profit_num' and c['c'] == 'test_1' and c['p'] == '2017: q1' and c['r'] == 'US'][0]['v']
        assert profit == 77.5
    with updated_settings({'adhoc_report_config': {
        'companies': ['test_1', 'test_2'],
        'lineitems': ['revenue_num', 'cost_num', 'profit_num'],
        'regions': ['US', 'WWR'],
        'display_rows': ['companies'],
        'display_columns': ['regions'],
        'periods_obj': [
            {'year': '2017', 'period': 'yr'},
            {'year': '2017', 'period': 'q1'},
        ]
    }, 'edited_cells': [{'c': 'test_1', 'l': 'cost_num', 'p': '2017: q1', 'r': 'US', 'v': 50}]}):
        cells = load_adhoc_report()['cells']
        assert len(cells) == 18
        profit = [c for c in cells if c['l'] == 'profit_num' and c['c'] == 'test_1' and c['p'] == '2017: q1' and c['r'] == 'US'][0]['v']
        assert profit == 50
        profit_wwr = [c for c in cells if c['l'] == 'profit_num' and c['c'] == 'test_1' and c['p'] == '2017: q1' and c['r'] == 'WWR'][0]['v']
        assert profit_wwr == 50
        assert [c for c in cells if c['l'] == 'cost_num' and c['c'] == 'test_1' and c['p'] == '2017: q1' and c['r'] == 'WWR'][0]['e'] is False

### end callables


class Cell:
    def __init__(self, field, value, input=False, error=None):
        self.field = field
        self.value = value
        self.input = input
        self.error = error
        self.rolled_up_from = {}

    def add_rollup(self, subthing, value):
        self.rolled_up_from[subthing] = value
        self.value += value

    def value_without_rollups(self, without_rollups):
        if self.value is None or len(without_rollups) == 0:
            return self.value
        return self.value - sum([self.rolled_up_from[subthing] for subthing in without_rollups if subthing in self.rolled_up_from])

    def __repr__(self):
        return ('<{}'.format(round(self.value, 2)) if self.value else '<null') + ':' + str(self.rolled_up_from)+'>'

    @property
    def for_ui(self):
        x = {
            'n': self.field,
            'v': self.value,
            'e': self.input,
            'c': []
        }
        if self.error:
            x['err'] = self.error
        return x


class ReportCalc:
    def __init__(self, company, year, period, currency, overrides=None):
        self.company = company
        self.year = year
        self.period = period
        self.currency = currency
        self.overrides = overrides if overrides else []
        self.load_report()

    @property
    def report_for_ui(self):
        for_ui = deepcopy(self.report)
        for region_obj in for_ui.values():
            for field, cell_obj in region_obj.items():
                region_obj[field] = cell_obj.for_ui
        return for_ui

    def load_report(self):
        # load it up
        report = get_report(self.company, self.year, self.period, self.currency)
        report = {region: line_item for region, line_item in report.items() if region.endswith('_obj')}
        for o in self.overrides:
            report[r2f(o['r'])][o['l']] = float(o['v'])
        report = drop_dynamic_fields(report, self.period)
        report = drop_non_leaf_fields(report)
        report = reg_fields_to_cells(report)
        self.report = geo_calc(report)

    def calc_report(self, simulate_dynamic_obj=None):
        """
        computes dynamic line item values for all regions
        :param field: report with all dynamic line items and non-leaf regions removed
        :return: report with rolled up values for ancestors of regions in input report, and all dynamic line items computed
        """
        dynamic_fields = sorted(get_dynamic_objs(self.period), key=dynamic_dependency_sorter)
        if simulate_dynamic_obj:  # TODO: bit hacky .. should just use an actual name [i.e. inject into dynamic line items; just pass dynamic_fields in instead of simulate_dynamic_obj]
            simulate_dynamic_obj['field'] = '_'
            dynamic_fields.append(simulate_dynamic_obj)
        errors = []
        for reg in self.report.keys():
            for dynamic_obj in dynamic_fields:
                ##### REQUIRED LINE ITEMS #####
                without_rollups = set()
                if 'req' in dynamic_obj:
                    rollups_can_do = None
                    all_rollups_seen = set()
                    for r in dynamic_obj['req']:
                        if r in self.report[reg]:
                            if self.report[reg][r].rolled_up_from.keys():
                                if rollups_can_do is None:
                                    rollups_can_do = set(self.report[reg][r].rolled_up_from.keys())
                                else:
                                    rollups_can_do = rollups_can_do.intersection(set(self.report[reg][r].rolled_up_from.keys()))
                                for ru in self.report[reg][r].rolled_up_from.keys():
                                    all_rollups_seen.add(ru)
                    if rollups_can_do is not None and len(rollups_can_do) != len(all_rollups_seen):
                        without_rollups = all_rollups_seen-rollups_can_do
                ##### / REQUIRED LINE ITEMS #####
                ns = {f.field: f.value_without_rollups(without_rollups) for f in self.report[reg].values()}
                ns['NotMeaningful'] = NotMeaningful
                for x in globals():
                    if x.startswith('Func_'):
                        ns[x[5:].upper()] = globals()[x](self.report[reg],
                                                         reg,
                                                         self.company,
                                                         self.year,
                                                         self.period,
                                                         self.currency)
                    if x.startswith('Const_'):
                        ns[x[6:].upper()] = globals()[x]
                err = None
                try:
                    if 'g' in dynamic_obj:
                        ns['G'] = dotdict()
                        exec(dynamic_obj['g'], ns)
                    v = eval(dynamic_obj['expr'][1:], ns)
                except NotMeaningful as e:
                    v = float(e.message)  # TODO should depend on field type
                    err = 'NotMeaningful'
                except TypeError:
                    continue  # TODO is that right?
                except Exception as e:
                    v = None
                    err = '{}: {}'.format(e.__class__.__name__, e)
                    #logger.exception(err)
                    #put stacktrace into cell TODO
                self.report[reg][dynamic_obj['field']] = Cell(field=dynamic_obj['field'], value=v, error=err)
                #except NameError as e:
                #    errors.append('Missing dependency: {}'.format(e.message)) #TODO errors
        if errors:
            logger.info(errors)
        return self.report

    def simulate_dynamic_obj(self, dynamic_obj):
        """Returns value for each region."""
        return {region_obj: by_li['_'].value for region_obj, by_li in self.calc_report(simulate_dynamic_obj=dynamic_obj).items() if '_' in by_li and by_li['_'].value}

    def simulate_dynamic_obj_for_ui(self, dynamic_obj):
        """"Returns value for each region in *for_ui* format."""
        return {region_obj: by_li['_'].for_ui for region_obj, by_li in self.calc_report(simulate_dynamic_obj=dynamic_obj).items() if '_' in by_li}


@sjutils.memoize
def get_report(company, year, period, currency='usd'):
    # TODO: Currently returns {'region_obj':{'li_num':12.34}}
    #       but should change to v_obj structure (array of {'v*','l','r'})
    query = r'sid:HFOD\Reports company="{}" year={} period={} currency={}'.format(company, year, period, currency)
    series = sjclient.get_series(query, fields=['*'])
    if series is None:
        return {}
    by_region = defaultdict(dict)
    for v in series['fields']['v_obj']:
        by_region[r2f(v['r'])][v['l']] = v['v_num']
    return by_region


def drop_dynamic_fields(report, period):
    """
    given report, finds and drops all dynamic line items
    :param report: report for given company, year, period
    :return: same report with all dynamic line items removed
    """
    # TODO: should only drop the ones dynamic for this period
    dynamic_fields = {d['field'] for d in get_dynamic_objs(period)}
    for key_1, val in report.items():
        if key_1.endswith('_obj'):  # then is region
            for key_2, val in report[key_1].items():
                if key_2 in dynamic_fields:
                    report[key_1].pop(key_2)
        else:
            if key_1 in dynamic_fields:
                report.pop(key_1)
    return report


def r2f(region):
    return '{}_obj'.format(region.lower())


def f2r(field):
    if not field.endswith('_obj'):
        raise Exception('not region field: {}'.format(field))
    return field[:-4].upper()


def drop_non_leaf_fields(report):
    """
    Accepts report which is {region_obj:{line_item:value}}
    Removes and line items from parent regions.  So only the very leaf region should have any given field.
    Ex. if both us_obj and noram_obj have a value for revenue_num, then noram_obj shoudln't have it any more after this
    function returns

    :param report: report containing region objects
    :return: same report with non-leaf line items removed from each region
    """
    unique_line_items = set()
    for line_items in report.values():
        for line_item in line_items:
            unique_line_items.add(line_item)

    region_names = set([f2r(field) for field in report.keys() if field.endswith('_obj')])
    region_names_with_descendants = {region_name: set(get_descendants(region_name)) for region_name in region_names}
    for line_item in unique_line_items:
        leaves = set()
        for region_name, descendants in region_names_with_descendants.items():
            region_descendants_with_values = set([rn for rn in descendants if report.get(r2f(rn), {}).get(line_item) is not None])
            # we know which regions this line item has values for... if the current region doesn't have any descendants
            # with a value, then we're a leaf!
            if len(region_descendants_with_values.intersection(region_names)) == 0:
                leaves.add(region_name)
        # strip the non-leaves
        for region_name, descendants in region_names_with_descendants.items():
            if region_name not in leaves:
                region = report[r2f(region_name)]
                if line_item in region:
                    region.pop(line_item)
    return report


def reg_fields_to_cells(report):
    """
    Given a report, converts all fieldname: fieldval field pairs to
    fieldname: {n: fieldname, v: fieldval, e: fieldeditable, c: fieldcomments} pairs.

    :param report: report to be converted (fieldname: fieldval format)
    :return: converted report (fieldname: {n: fieldname, v: fieldval, e: fieldeditable, c: fieldcomments} format)
    """
    report_copy = report.copy()
    for key in report_copy.keys():
        if key.endswith('_obj'):
            reg_fields = report.pop(key)
            field_entries = {}
            for key_2, val in reg_fields.items():
                field_entries[key_2] = Cell(value=val, input=True, field=key_2)
            report[key] = field_entries
    return report


def geo_calc(editable_fields):
    """
    given report with all dynamic line items and non-leaf regions removed, roll up values for non-leaf regions
    :param editable_fields: report with all dynamic line items and non-leaf regions removed
    :return: report with rolled up values for ancestors of regions in input report
    """

    output = editable_fields
    for line_item in set([field for reg in editable_fields for field in editable_fields[reg]]):
        leaf_keys = [field for field in editable_fields.keys() if field.endswith('_obj') and line_item in editable_fields[field]]
        leaves = set([f2r(leaf_key) for leaf_key in leaf_keys])
        output_regs = set()
        for leaf in leaves:
            output_regs.update(get_ancestors(leaf))
        for reg_name in output_regs:
            reg_keyname = r2f(reg_name)
            if reg_keyname not in output:
                output[reg_keyname] = {}
            if line_item in editable_fields[reg_keyname]:
                raise Exception("ERROR - can not have dependent geographies")
            output[reg_keyname][line_item] = Cell(value=0, field=line_item)
            # for data to be used in computation, must be descendant of output region and be a leaf region
            reg_descs = leaves.intersection(get_descendants(reg_name))
            for desc_name in reg_descs:
                desc_keyname = r2f(desc_name)
                output[reg_keyname][line_item].add_rollup(desc_keyname, editable_fields[desc_keyname][line_item].value)
    return output


class Func:
    def __init__(self, region_obj, region_name, company, year, period, currency='usd'):
        self.region_obj = region_obj
        self.region_name = region_name
        self.company = company
        self.year = year
        self.period = period
        self.currency = currency

    def get_relative_year_period(self, year=0, period=0):
        """Returns (year, period) based on following logic:
        year -- default 0; can be absolute year or relative (0 for current; -1 for last year)
        period -- can be one of 7 periods: yr,h1,m9,q1-4 or relative (0 for current; -1 for last period)
        """
        absolute_periods = ['yr', 'h1', 'm9', 'q1', 'q2', 'q3', 'q4']
        year_out = int(self.year)
        period_out = self.period
        if year > 50:  # absolute year
            year_out = year
        else:  # relative year
            year_out += year
        if period in absolute_periods:  # absolute period
            period_out = period
        else:  # relative period
            if self.period in ('yr', 'm9', 'h1'):  # period is same thing as year
                year_out += period
            else:  # period is quarterly
                my_zero_quarter = int(self.period[1])-1
                zero_quarter_diff = my_zero_quarter+period
                new_zero_quarter = zero_quarter_diff % 4
                year_out += int((zero_quarter_diff if zero_quarter_diff >= 0 else zero_quarter_diff-4)/4.0)
                period_out = 'q{}'.format(new_zero_quarter+1)
        return year_out, period_out

    def f(self, field, year=0, period=0, ifnull=None, region=None):
        if region is None:
            region = self.region_name
        year, period = self.get_relative_year_period(year, period)
        if year == self.year and period == self.period and region == self.region_name:
            return self.region_obj[field].value if field in self.region_obj else ifnull
        else:
            return get_report(self.company, year, period, self.currency).get(region, {}).get(field, ifnull)

    def __call__(self, whatever, param):
        raise Exception('not implemented')


Const_RRC_LIMIT = 60.0
Const_PROD_REPLACEMENT_LIMIT = 10.0


class Func_f(Func):
    def __call__(self, field, year=0, period=0, ifnull=None):
        """
        Return field value in specified period.
        DOES NOT RE-RUN CALCULATIONS.  JUST GRABS STATIC NUMBERS.

        field::name of field
        year::default use current year; if period=="yr" becomes previous year; if <=50 treat like n_years_back; if period is int, subject to calculation
        period::default previous period; if str, get period from specified year; if int, use as lookback from current period
        ifnull::value to return if result is None
        """
        return self.f(field, year, period, ifnull)


class Func_esum(Func):
    def __call__(self, adders=None, subtractors=None):
        """
        TODO: improve docs: adds up line_items.  if all the things are null, returns null; if any things are there;
              returns the things
        """
        if adders:
            add = [self.f(li) for li in adders]
            add = [v for v in add if v is not None]
        else:
            add = []
        if subtractors:
            sub = [self.f(li) for li in subtractors]
            sub = [v*-1.0 for v in sub if v is not None]
        else:
            sub = []
        combine = add+sub
        if combine:
            return sum(combine)
        else:
            return None


class Func_trail(Func):
    def __call__(self, field, n_years_back=1, method='sum'):
        """
        Returns [method] of field value for all periods in range.  Null values are counted as 0.

        field::name of field
        n_years_back::inclusive year range; i.e. if it's 2018 and n_years_back==2, 2016 is the beginning year
        method::"sum", "avg" or "med"
        """
        values = []
        for yr_back in range(n_years_back + 1):
            values.append(self.f(field, year=-yr_back, ifnull=0))
        if method == 'sum':
            return sum(values)
        elif method == 'avg':
            return sum(values) / len(values)
        elif method == 'med':
            return sorted(values)[int(len(values)/2)]


class Func_cagr(Func):
    def __call__(self, field, n_years_back=1):
        """
        Returns compound annual growth rate over year range, defined by equation:
            (Ending Value / Beginning Value) ** (1 / number of trailing years) - 1

        field::name of field
        n_years_back::inclusive year range; i.e. if it's 2018 and n_years_back==2, 2016 is the beginning year
        """
        v0 = self.f(field, year=-n_years_back)
        v1 = self.f(field)
        if v0 is None or v1 is None:
            return None
        return (v1 / v0) ** (1.0 / n_years_back) - 1


class Func_pop(Func):
    def __call__(self, field, n_periods_back=1):
        """
        Returns compound annual growth rate over year range, defined by equation:
            (Ending Value / Beginning Value) ** (1 / number of trailing years) - 1

        field::name of field
        n_years_back::inclusive year range; i.e. if it's 2018 and n_years_back==2, 2016 is the beginning year
        """
        v0 = self.f(field, period=-n_periods_back)
        v1 = self.f(field)
        if v0 is None or v1 is None:
            return None
        return v1/v0-1.0


class Func_current_period_days(Func):
    def __call__(self):
        """
        Returns number of days in current period.  Based on wwf/gi_fiscalpe_num.
        """
        year_end = self.year
        month_end = self.f('gi_fiscalpe_num', region='wwf_obj')
        period_to_months = {
            'q1': 3,
            'q2': 3,
            'q3': 3,
            'q4': 3,
            'yr': 12,
            'm9': 9,
            'h1': 6,
        }
        df = date(self.year, month_end, 1)-relativedelta(months=period_to_months[self.period]-1)
        dt = date(self.year, month_end, 1)+relativedelta(months=1)-relativedelta(days=1)
        return 1+(dt-df).days


class Func_TTM(Func):
    def __call__(self, field, ifnull=None):
        """
        TTM. Examples:
        2018Q1 = 2018Q1 + (2017YR - 2017M9) + 2017Q3 + 2017Q2
        2018Q2 = 2018Q2+2018Q1+(2017YR - 2017M9)+2017Q3
        2018Q3 = 2018Q3+2018Q2+2018Q1+(2017YR - 2017M9)
        """
        if self.period not in ('q1', 'q2', 'q3'):
            raise Exception('TTM can only be used on periods q1-q3')
        prev_q4 = self.f(field, year=-1, ifnull=ifnull) - self.f(field, year=-1, period='m9', ifnull=ifnull)
        if self.period == 'q1':
            return self.f(field, ifnull=ifnull) + prev_q4 + self.f(field, year=-1, period='q3', ifnull=ifnull) + self.f(field, year=-1, period='q2', ifnull=ifnull)
        elif self.period == 'q2':
            return self.f(field, ifnull=ifnull) + self.f(field, period='q1', ifnull=ifnull) + prev_q4 + self.f(field, year=-1, period='q3', ifnull=ifnull)
        elif self.period == 'q3':
            return self.f(field, ifnull=ifnull) + self.f(field, period='q2', ifnull=ifnull) + self.f(field, period='q1', ifnull=ifnull) + prev_q4


def get_family_tree(geo_name):
    """Returns dict-based family tree under node geo_name. Relies on node_dict"""
    return {child.lower(): get_family_tree(child) for child in GEO[geo_name].get('children', [])}


def periodize_dynamic_obj(dynamic_obj, period):
    expr = dynamic_obj.get('{}_expr'.format(period), dynamic_obj.get('expr'))
    if expr is None:
        return None
    o = {
        'dd': dynamic_obj.get('{}_dd'.format(period), dynamic_obj.get('dd')),
        'g': dynamic_obj.get('{}_g_expr'.format(period), dynamic_obj.get('g')),
        'expr': expr,
    }
    o = {k: v for k, v in o.items() if v is not None}
    return o


@sjutils.memoize
def get_dynamic_objs(period):
    """
    finds all dynamic line items, returns array IN ORDER of dynamic dependencies with keys: dd, g, expr, field
    :return: dictionary with fieldname:expression pairs
    """
    dynamic_objs = []
    for series in sjclient.scroll('sid:HFOD\\LineItems set=dynamic_obj ', fields=['dynamic_obj']):
        sid = series['series_id']
        fieldname = sid.split('\\')[-1]
        o = periodize_dynamic_obj(series['fields']['dynamic_obj'], period)
        if o is not None:
            o['field'] = fieldname
            dynamic_objs.append(o)
    return dynamic_objs


def dynamic_dependency_sorter(a):
    # TODO: this is retarded... we need a/b comparison and figure out if one should run before other
    if 'dd' in a:
        return len(''.join(a['dd']))
    return 0


def get_descendants(node_name, desc_list=None):
    """
    given the name of a node, get all of its descendants
    :param node_name: name of node
    :param desc_list: list of descendants that are already found. only used in recursive calls
    :return: list of all descendants of node with name nodename
    """
    if desc_list is None:
        desc_list = []
    if node_name not in GEO:  # TODO: is this ok?
        return []
    node = GEO[node_name]
    for child in node.get('children', []):
        desc_list.append(child)
        desc_list = get_descendants(child, desc_list)
    return desc_list


def get_ancestors(node_name, anc_list=None):
    """
    given the name of a node, get all of its ancestors
    :param node_name: name of node
    :param anc_list: list of ancestors that are already found. only used in recursive calls
    :return: list of all ancestors of node with name nodename
    """
    if anc_list is None:
        anc_list = []
    if node_name not in GEO:  # TODO: is this ok?
        return []
    node = GEO[node_name]
    parent = node.get('parent')
    if parent:
        if isinstance(parent, list):  # for (rare) case that child has more than one parent
            for p in parent:
                anc_list.append(p)
                anc_list += get_ancestors(p, anc_list)
        else:  # single parent case
            anc_list.append(parent)
            anc_list = get_ancestors(parent, anc_list)

    return anc_list


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


###############################
############ TESTS ############
###############################


def test():
    number_of_big_tests = 0
    for name, func in globals().items():
        if name.startswith('TEST_'):
            number_of_big_tests += 1
            try:
                func()
                logger.debug('{} success!'.format(name))
            except:
                logger.exception('{} failed'.format(name))
    if number_of_big_tests > 0:
        logger.debug('skipping tests because ran {} big tests'.format(number_of_big_tests))
        return True
    for name, func in globals().items():
        if name.startswith('test_'):
            try:
                func()
                logger.debug('{} success!'.format(name))
            except:
                logger.exception('{} failed'.format(name))
    return True


@contextmanager
def updated_settings(s):
    global settings
    original_settings = deepcopy(settings)
    settings.update(s)
    yield
    settings = original_settings


def test_get_report():
    assert get_report('test_1', '2017', 'yr')['us_obj']['revenue_num'] == 100.0, r'Query "sid:HFOD\Reports company="test_1" year=2017 period=yr currency=usd" should return us_obj.revenue_num'


def test_esum():
    rc = ReportCalc('test_1', 2017, 'yr', 'usd')
    assert rc.simulate_dynamic_obj({
        'expr': "=ESUM(['xxx'])"
    }) == {}
    assert rc.simulate_dynamic_obj({
        'expr': "=ESUM(['revenue_num'])"
    })['us_obj'] == 100.0
    assert rc.simulate_dynamic_obj({
        'expr': "=ESUM(['revenue_num','xxx'])"
    })['us_obj'] == 100.0
    assert rc.simulate_dynamic_obj({
        'expr': "=ESUM(['revenue_num','profit_num'])"
    })['us_obj'] == 194.0
    assert rc.simulate_dynamic_obj({
        'expr': "=ESUM(['revenue_num','profit_num','xxx'])"
    })['us_obj'] == 194.0
    assert rc.simulate_dynamic_obj({
        'expr': "=ESUM(['revenue_num'],['profit_num','xxx'])"
    })['us_obj'] == 6.0


def test_trail():
    rc = ReportCalc('test_1', 2017, 'yr', 'usd')
    assert rc.simulate_dynamic_obj({
        'expr': "=TRAIL('revenue_num', 2)"
    }) == {'regroll_obj': 150.0, 'us_obj': 281.0, 'wwr_obj': 150.0, 'can_obj': 150.0, 'con_obj': 150.0, 'regwest_obj': 150.0, 'noram_obj': 150.0}
    assert round(rc.simulate_dynamic_obj({
        'expr': "=TRAIL('revenue_num', 2, 'avg')"
    })['us_obj'], 2) == 93.67
    assert rc.simulate_dynamic_obj({
        'expr': "=TRAIL('revenue_num', 2, 'med')"
    })['us_obj'] == 91.0
    rc = ReportCalc('test_1', 2017, 'q4', 'usd')
    assert rc.simulate_dynamic_obj({
        'expr': "=TRAIL('revenue_num', 1)"
    })['us_obj'] == 47.5


def test_cagr():
    rc = ReportCalc('test_1', 2017, 'yr', 'usd')
    assert round(rc.simulate_dynamic_obj({
        'expr': "=POP('revenue_num', 1)"
    })['us_obj'], 2) == 0.11
    assert round(rc.simulate_dynamic_obj({
        'expr': "=POP('revenue_num')"
    })['us_obj'], 2) == 0.11


def test_pop():
    rc = ReportCalc('test_1', 2017, 'yr', 'usd')
    assert round(rc.simulate_dynamic_obj({
        'expr': "=CAGR('revenue_num', 2)"
    })['us_obj'], 2) == 0.05
    assert round(rc.simulate_dynamic_obj({
        'expr': "=CAGR('revenue_num')"
    })['us_obj'], 2) == 0.11


def test_get_full_report_with_rollups_and_calc():
    with updated_settings({'company': 'test_1', 'year': '2017', 'period': 'yr'}):
        r = get_full_report()
        assert r['noram_obj']['revenue_num']['v'] == 150.0, r'Should rollup US and Canada for sid=HFOD\Reports\test_1\2017\yr\usd'
        assert r['noram_obj']['profit_num']['v'] == 138.0, 'NORAM net profit'


def test_get_relative_year_period():
    f = Func({}, 'US', 'test_1', 2017, 'yr')
    assert f.get_relative_year_period() == (2017, 'yr'), 'no change'
    assert f.get_relative_year_period(-5) == (2012, 'yr'), '5 yrs ago'
    assert f.get_relative_year_period(-5, -2) == (2010, 'yr'), '7 yrs ago'
    assert f.get_relative_year_period(-5, 'h1') == (2012, 'h1'), 'set h1'
    assert f.get_relative_year_period(2016, 'h1') == (2016, 'h1'), 'very absolute'
    f = Func({}, 'US', 'test_1', 2017, 'q2')
    assert f.get_relative_year_period(2016) == (2016, 'q2'), 'no change'
    assert f.get_relative_year_period(2016, -1) == (2016, 'q1'), 'prev quarter'
    assert f.get_relative_year_period(2016, -2) == (2015, 'q4'), 'prev quarter'
    assert f.get_relative_year_period(2016, -3) == (2015, 'q3'), 'prev quarter'
    assert f.get_relative_year_period(2016, -4) == (2015, 'q2'), 'prev quarter'
    assert f.get_relative_year_period(2016, -8) == (2014, 'q2'), 'prev quarter'
    assert f.get_relative_year_period(2016, -18) == (2011, 'q4'), 'prev quarter'
    assert f.get_relative_year_period(2016, 0) == (2016, 'q2'), 'this quarter'
    assert f.get_relative_year_period(2016, 1) == (2016, 'q3'), 'next quarter'
    assert f.get_relative_year_period(2016, 2) == (2016, 'q4'), 'next quarter'
    assert f.get_relative_year_period(2016, 3) == (2017, 'q1'), 'next quarter'


def test_f():
    rc = ReportCalc('test_1', 2017, 'yr', 'usd')
    assert rc.simulate_dynamic_obj({
        'expr': "=F('revenue_num')"
    }) == {'regroll_obj': 150, 'us_obj': 100.0, 'wwr_obj': 150.0, 'can_obj': 50.0, 'con_obj': 150.0, 'regwest_obj': 150.0, 'noram_obj': 150.0}
    assert rc.simulate_dynamic_obj({
        'expr': "=F('revenue_num', period=-1)"
    }) == {'can_obj': 50.0, 'us_obj': 90.0}, 'rollups are not done by default for previous years'


def test_g():
    rc = ReportCalc('test_1', 2017, 'yr', 'usd')
    assert rc.simulate_dynamic_obj({
        'expr': "=revenue_num*G.m",
        'g': 'G.m = 10'
    })['us_obj'] == 1000


def test_dynamic_dependencies():
    rc = ReportCalc('test_1', 2017, 'yr', 'usd')
    assert rc.calc_report()['us_obj']['double_double_profit_num'].value == 376.0, 'double double!'


def test_required_line_item():
    rc = ReportCalc('test_1', 2015, 'yr', 'usd')
    assert round(rc.calc_report()['noram_obj']['revenue_over_cost_num'].value, 2) == 0.98, 'canada is excluded...'


def test_errors():
    rc = ReportCalc('test_1', 2017, 'yr', 'usd')
    assert rc.simulate_dynamic_obj_for_ui({
        'expr': "=revenue_num*xx",
    })['us_obj']['err'] == "NameError: name 'xx' is not defined"


def test_get_geo_tree():
    assert 'tree' in get_geo_tree()


def XXX_test_preview_expression():
    with updated_settings({
        "dynamic_obj": {"expr": "=RO_CF_GROWTH_NUM*2", "g_expr": ""},
        "visibility_status": "Visible To Clients",
        "region": "non_geog",
        "company": "AE",
        "periodicity": "yr",
        "notes": "",
        "dynamic_req_dependencies": None
    }):
        res = preview_expression()
        logger.debug(res)


def test_ttm():
    rc = ReportCalc('test_1', 2017, 'q1', 'usd')
    assert rc.simulate_dynamic_obj({
        'expr': "=TTM('revenue_num', ifnull=0)"
    })['wwr_obj'] == 100
    return  # why next?
    assert 'wwr_obj' not in rc.simulate_dynamic_obj({
        'expr': "=TTM('revenue_num')"
    })


def test_constants():
    rc = ReportCalc('test_1', 2017, 'yr', 'usd')
    assert rc.simulate_dynamic_obj({
        'expr': "=RRC_LIMIT"
    })['wwr_obj'] == 60
    assert rc.simulate_dynamic_obj({
        'expr': "=PROD_REPLACEMENT_LIMIT"
    })['wwr_obj'] == 10


def test_current_period_days():
    rc = ReportCalc('test_1', 2017, 'yr', 'usd')
    assert rc.simulate_dynamic_obj({
        'expr': "=CURRENT_PERIOD_DAYS()"
    })['wwr_obj'] == 365

    rc = ReportCalc('test_1', 2016, 'yr', 'usd')
    assert rc.simulate_dynamic_obj({
        'expr': "=CURRENT_PERIOD_DAYS()"
    })['wwr_obj'] == 366

    rc = ReportCalc('test_1', 2017, 'q4', 'usd')
    assert rc.simulate_dynamic_obj({
        'expr': "=CURRENT_PERIOD_DAYS()"
    })['wwr_obj'] == 91


# --- GEO ---

geo_yaml = '''
AFR:
  name: West Africa
  parent: AFRME
AFRME:
  children:
  - AFR
  - AFRNO
  - MIDLEAST
  - MID
  name: Africa/Middle East
  parent: REGEAST
AFRNO:
  name: North Africa
  parent: AFRME
AGGCON:
  name: Aggregate Consol. Only
  parent: CON
AGGNCI:
  name: Aggregate Non-Consol. Only
  parent: NCI
AGGRD:
  name: Aggregate Consol. & Non-Consol.
  parent: WWR
AGGRUCA:
  name: Aggregate Russia/Caspian
  parent: RUSCASP
ARGEN:
  name: Argentina
  parent: LATAM
ASPAC:
  children:
  - ESASPAC
  - RUSCASP
  - OTHSPA
  name: Asia-Pacific
  parent: REGEAST
AUL:
  name: Australia
  parent: ESASPAC
BOLIVIA:
  name: Bolivia
  parent: LATAM
BRAZIL:
  name: Brazil
  parent: LATAM
CAN:
  children:
  - CANNET
  - CANSYN
  - CANBIT
  - CANAGGR
  name: 'Canada (Consol. Only): Total, net'
  parent: NORAM
CANAGGR:
  name: 'Canada: Aggregate, net'
  parent: CAN
CANBIT:
  name: 'Canada: Bitumen, net'
  parent: CAN
CANCNCI:
  name: 'Canada: Total (Consol. & Non-Consol.)'
  parent: WWR
CANNET:
  name: 'Canada: Conventional, net'
  parent: CAN
CANSYN:
  name: 'Canada: Synthetic Oil, net'
  parent: CAN
CASPIAN:
  name: Caspian Region
  parent: RUSCASP
CHN:
  name: China
  parent: ESASPAC
COL:
  name: Colombia
  parent: LATAM
CON:
  children:
  - REGWEST
  - REGEAST
  - OTH
  - AGGCON
  name: Worldwide (Consolidated only)
  parent: WWR
EASTEUR:
  name: Eastern Europe
  parent: EUR
ECUADOR:
  name: Ecuador
  parent: LATAM
ESASPAC:
  children:
  - AUL
  - CHN
  - INDIASUB
  - INDO
  - THAILAND
  - SOUEAS
  name: East & South Asia-Pacific
  parent: ASPAC
EUR:
  children:
  - NSE
  - EASTEUR
  - OTHEUR
  name: 'Europe: Total (Consolidated only)'
  parent: REGEAST
EUROCNCI:
  name: 'Europe: Total (Consol. & Non-Consol.)'
  parent: WWR
GRSAGG:
  name: Gross Aggregate
GRSCAGG:
  name: 'Gross Canada: Aggregate'
  parent: GRSCTOT
GRSCBIT:
  name: 'Gross Canada: Bitumen'
  parent: GRSCTOT
GRSCCON:
  name: 'Gross Canada: Conventional'
  parent: GRSCTOT
GRSCSYN:
  name: 'Gross Canada: Synthetic Oil'
  parent: GRSCTOT
GRSCTOT:
  children:
  - GRSCCON
  - GRSCSYN
  - GRSCBIT
  - GRSCAGG
  name: Gross Canada Total
GRSOTH:
  name: Gross Other/Unspecified
GRSUS:
  name: Gross United States
GRSWW:
  name: 'Gross Canada Supplement: Worldwide'
INDIASUB:
  name: Indian Sub-Continent
  parent: ESASPAC
INDO:
  name: Indonesia
  parent: ESASPAC
LATAM:
  children:
  - ARGEN
  - BOLIVIA
  - BRAZIL
  - COL
  - ECUADOR
  - MEXICO
  - TRINID
  - VENTOTAL
  - OTHSOUAM
  name: South & Central America
  parent: REGWEST
MEXICO:
  name: Mexico
  parent: LATAM
MID:
  name: Other/Aggr. Africa/Middle East
  parent: AFRME
MIDLEAST:
  name: Middle East
  parent: AFRME
NACNCI:
  name: North America (Consol. & Non-Consol.)
  parent: WWR
NCI:
  children:
  - NCIUS
  - NCICAN
  - NCIRUCAS
  - NCIEURO
  - NCI1
  - NCI2
  - AGGNCI
  name: 'Non-Consolidated (NCI): Total'
  parent: WWR
NCI1:
  name: 'Non-Consolidated: Non-U.S.1'
  parent: NCI
NCI2:
  name: 'Non-Consolidated: Non-U.S.2'
  parent: NCI
NCICAN:
  name: 'Non-Consolidated: Canada'
  parent: NCI
NCIEURO:
  name: 'Non-Consolidated: Europe'
  parent: NCI
NCIRUCAS:
  name: 'Non-Consolidated: Russia & Caspian'
  parent: NCI
NCIUS:
  children:
  - USCNCI
  name: 'Non-Consolidated: U.S.'
  parent: NCI
NORAM:
  children:
  - US
  - CAN
  name: 'North America: U.S. & Canada (Consol. Only)'
  parent: REGWEST
NSE:
  name: North Sea/Northern Europe
  parent: EUR
OEH:
  name: Other Eastern Hemisphere
  parent: REGEAST
OTH:
  name: Other/Unspecified
  parent: CON
OTHEUR:
  name: Other Europe
  parent: EUR
OTHSOUAM:
  name: Other/Unspec. Central/Sou. Amer.
  parent: LATAM
OTHSPA:
  name: Other/Aggr. Asia-Pacific
  parent: ASPAC
OWH:
  name: Other Western Hemisphere
  parent: REGWEST
REGEAST:
  children:
  - AFRME
  - ASPAC
  - EUR
  - OEH
  name: Eastern Hemisphere
  parent: CON
REGWEST:
  children:
  - NORAM
  - LATAM
  - OWH
  name: Western Hemisphere
  parent: CON
RUCACNCI:
  name: 'Russia & Caspian: Total (Consol. & NCI)'
  parent: WWR
RUS:
  name: Russia
  parent: RUSCASP
RUSCASP:
  children:
  - RUS
  - CASPIAN
  - AGGRUCA
  name: Russia & Caspian (Consol. only)
  parent: ASPAC
SOUEAS:
  name: Other/Aggregate East & South Asia
  parent: ESASPAC
THAILAND:
  name: Thailand
  parent: ESASPAC
TRINID:
  name: Trinidad & Tobago
  parent: LATAM
REGROLL:
  name: All Regions
  children:
  - WWF
  - WWR
US:
  children:
  - USCNCI
  name: U.S. (Consolidated only)
  parent: NORAM
USCNCI:
  name: U.S. (Consol. & Non-Consol.)
  parent:
  - US
  - NCIUS
VENCONV:
  name: 'Venezuela: Conventional'
  parent: VENTOTAL
VENTOTAL:
  children:
  - VENCONV
  - VENXHEAV
  name: 'Venezuela: Total'
  parent: LATAM
VENXHEAV:
  name: 'Venezuela: Extra Heavy Crude'
  parent: VENTOTAL
WONA:
  name: World Outside North America
  parent: WWR
WOUS:
  name: World Outside U.S. (Consol. & NCI)
  parent: WWR
WOUSCON:
  name: World Outside U.S. Consol. Only
  parent: WWR
WWF:
  name: Financial Statements (R)
  parent: REGROLL
WWR:
  parent: REGROLL
  children:
  - CON
  - NCI
  - AGGRD
  name: Worldwide (Consol. & NCI)
  previous_kids:
  - NACNCI
  - WONA
  - WOUS
  - WOUSCON
  - USCNCI
  - CANCNCI
  - RUCACNCI
  - EUROCNCI
'''
GEO = yaml.load(geo_yaml, Loader=yaml.FullLoader)
