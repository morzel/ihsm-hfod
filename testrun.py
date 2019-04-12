import hfod_calc as hfod
import contextlib
import copy
import unittest

TEST_REPORTS = {
    'test_1.2017.yr.usd': {
        'us_obj': {
            'profit_num': 94.0,
            'revenue_num': 100.0,
            'cost_num': 6,
        },
        'can_obj': {
            'revenue_num': 50.0,
            'cost_num': 6,
        },
        'wwf_obj': {
            'gi_fiscalpe_num': 10
        }
    },
    'test_2.2017.yr.usd': {
        'us_obj': {
            'revenue_num': 120.0,
            'cost_num': 5,
        },
        'can_obj': {
            'revenue_num': 50.0,
            'cost_num': 7,
        }
    },
    'test_3.2017.yr.usd': {
        'noram_obj': {
            'revenue_num': 50.0,
            'cost_num': 50,
        }
    },
    'test_1.2016.yr.usd': {
        'us_obj': {
            'revenue_num': 90.0,
            'cost_num': 89.0,
        },
        'can_obj': {
            'revenue_num': 50.0,
            'cost_num': 6,
        },
        'wwf_obj': {
            'gi_fiscalpe_num': 10
        }
    },
    'test_1.2015.yr.usd': {
        'us_obj': {
            'revenue_num': 91.0,
            'cost_num': 93.0,
        },
        'can_obj': {
            'revenue_num': 50.0,
        }
    },
    'test_1.2017.q1.usd': {
        'us_obj': {
            'revenue_num': 100.0,
            'cost_num': 22.5,
            'profit_num': 77.5
        }
    },
    'test_1.2017.q4.usd': {
        'us_obj': {
            'revenue_num': 25,
            'cost_num': 22.5,
        },
        'wwf_obj': {
            'gi_fiscalpe_num': 11
        }
    },
    'test_1.2016.q4.usd': {
        'us_obj': {
            'revenue_num': 22.5,
            'cost_num': 22.5,
        }
    }
}


def get_report(company, year, period, currency='usd'):
    key = '{}.{}.{}.{}'.format(company, year, period, currency)
    return TEST_REPORTS.get(key, {})


def get_dynamic_objs(_):
    return [
        {
            'field': 'double_double_profit_num',
            'expr': '=double_profit_num*2',
            'dd': ['double_profit_num']
        },
        {
            'field': 'double_profit_num',
            'expr': '=profit_num*2',
            'dd': ['profit_num']
        },
        {
            'field': 'revenue_over_cost_num',
            'expr': '=revenue_num/cost_num',
            'req': ['revenue_num', 'cost_num']
        },
        {
            'field': 'profit_num',
            'expr': '=revenue_num-cost_num',
        }
    ]


def get_companies():
    return [{
        'id': 'test_1',
        'peer_groups': [
            'group_a',
            'group_b'
        ]
    }, {
        'id': 'test_2',
        'peer_groups': [
            'group_a',
            'group_c'
        ]
    }, {
        'id': 'test_3',
        'peer_groups': [
            'group_a',
            'group_b'
        ]
    }]


# Old tests moved from hfod_calc.py.
class TestHfodCalcModule(unittest.TestCase):

    def test_cagr(self):
        rc = hfod.ReportCalc('test_1', 2017, 'yr', 'usd')
        self.assertAlmostEqual(rc.simulate_dynamic_obj({'expr': "=CAGR('revenue_num', 2)"})['us_obj'], 0.05, places=2)
        self.assertAlmostEqual(rc.simulate_dynamic_obj({'expr': "=CAGR('revenue_num')"})['us_obj'], 0.11, places=2)

    def test_constants(self):
        rc = hfod.ReportCalc('test_1', 2017, 'yr', 'usd')
        self.assertEqual(rc.simulate_dynamic_obj({'expr': "=RRC_LIMIT"})['wwr_obj'], 60)
        self.assertEqual(rc.simulate_dynamic_obj({'expr': "=PROD_REPLACEMENT_LIMIT"})['wwr_obj'], 10)

    def test_current_period_days(self):
        rc = hfod.ReportCalc('test_1', 2017, 'yr', 'usd')
        self.assertEqual(rc.simulate_dynamic_obj({'expr': "=CURRENT_PERIOD_DAYS()"})['wwr_obj'], 365)

        rc = hfod.ReportCalc('test_1', 2016, 'yr', 'usd')
        self.assertEqual(rc.simulate_dynamic_obj({'expr': "=CURRENT_PERIOD_DAYS()"})['wwr_obj'], 366)

        rc = hfod.ReportCalc('test_1', 2017, 'q4', 'usd')
        self.assertEqual(rc.simulate_dynamic_obj({'expr': "=CURRENT_PERIOD_DAYS()"})['wwr_obj'], 91)

    def test_dynamic_dependencies(self):
        rc = hfod.ReportCalc('test_1', 2017, 'yr', 'usd')
        self.assertEqual(rc.calc_report()['us_obj']['double_double_profit_num'].value, 376.0, msg='double double!')

    def test_errors(self):
        rc = hfod.ReportCalc('test_1', 2017, 'yr', 'usd')
        self.assertEqual(rc.simulate_dynamic_obj_for_ui({'expr': "=revenue_num*xx"})['us_obj']['err'],
                         "NameError: name 'xx' is not defined")

    def test_esum(self):
        rc = hfod.ReportCalc('test_1', 2017, 'yr', 'usd')
        self.assertEqual(rc.simulate_dynamic_obj({'expr': "=ESUM(['xxx'])"}), {})
        self.assertEqual(rc.simulate_dynamic_obj({'expr': "=ESUM(['revenue_num'])"})['us_obj'], 100.0)
        self.assertEqual(rc.simulate_dynamic_obj({'expr': "=ESUM(['revenue_num','xxx'])"})['us_obj'], 100.0)
        self.assertEqual(rc.simulate_dynamic_obj({'expr': "=ESUM(['revenue_num','profit_num'])"})['us_obj'], 194.0)
        self.assertEqual(rc.simulate_dynamic_obj({'expr': "=ESUM(['revenue_num','profit_num','xxx'])"})['us_obj'],
                         194.0)
        self.assertEqual(rc.simulate_dynamic_obj({'expr': "=ESUM(['revenue_num'],['profit_num','xxx'])"})['us_obj'],
                         6.0)

    def test_f(self):
        rc = hfod.ReportCalc('test_1', 2017, 'yr', 'usd')
        self.assertEqual(
            rc.simulate_dynamic_obj({'expr': "=F('revenue_num')"}),
            {
                'regroll_obj': 150,
                'us_obj': 100.0,
                'wwr_obj': 150.0,
                'can_obj': 50.0,
                'con_obj': 150.0,
                'regwest_obj': 150.0,
                'noram_obj': 150.0
            }
        )
        self.assertEqual(
            rc.simulate_dynamic_obj({'expr': "=F('revenue_num', period=-1)"}),
            {'can_obj': 50.0, 'us_obj': 90.0},
            msg='rollups are not done by default for previous years'
        )

    def test_g(self):
        rc = hfod.ReportCalc('test_1', 2017, 'yr', 'usd')
        self.assertEqual(rc.simulate_dynamic_obj({'expr': "=revenue_num*G.m", 'g': 'G.m = 10'})['us_obj'], 1000)

    def test_get_full_report_with_rollups_and_calc(self):
        with self.updated_settings({'company': 'test_1', 'year': '2017', 'period': 'yr'}):
            r = hfod.get_full_report()
            self.assertEqual(r['noram_obj']['revenue_num']['v'],
                             150.0,
                             msg=r'Should rollup US and Canada for sid=HFOD\Reports\test_1\2017\yr\usd')
            self.assertEqual(r['noram_obj']['profit_num']['v'],
                             138.0,
                             msg='NORAM net profit')

    def test_get_geo_tree(self):
        self.assertIn('tree', hfod.get_geo_tree())

    def test_get_report(self):
        self.assertEqual(hfod.get_report('test_1', '2017', 'yr')['us_obj']['revenue_num'],
                         100.0,
                         msg=r'Query "sid:HFOD\Reports company="test_1" year=2017 period=yr currency=usd" should'
                             r' return us_obj.revenue_num')

    def test_load_adhoc_report(self):
        test_settings = {
            'adhoc_report_config': {
                'companies': ['test_1', 'test_2'],
                'lineitems': ['revenue_num', 'cost_num', 'profit_num'],
                'regions': ['US', 'WWR'],
                'display_rows': ['companies'],
                'display_columns': ['regions'],
                'periods_obj': [{'year': '2017', 'period': 'yr'}, {'year': '2017', 'period': 'q1'}]
            }
        }
        with self.updated_settings(test_settings):
            cells = hfod.load_adhoc_report()['cells']
            self.assertEqual(len(cells), 18)
            profit = None
            for c in cells:
                if c['l'] == 'profit_num' and c['c'] == 'test_1' and c['p'] == '2017: q1' and c['r'] == 'US':
                    profit = c['v']
            self.assertEqual(profit, 77.5)
        test_settings['edited_cells'] = [{'c': 'test_1', 'l': 'cost_num', 'p': '2017: q1', 'r': 'US', 'v': 50}]
        with self.updated_settings(test_settings):
            cells = hfod.load_adhoc_report()['cells']
            self.assertEqual(len(cells), 18)
            profit = None
            for c in cells:
                if c['l'] == 'profit_num' and c['c'] == 'test_1' and c['p'] == '2017: q1' and c['r'] == 'US':
                    profit = c['v']
            self.assertEqual(profit, 50)
            profit_wwr = None
            for c in cells:
                if c['l'] == 'profit_num' and c['c'] == 'test_1' and c['p'] == '2017: q1' and c['r'] == 'WWR':
                    profit_wwr = c['v']
            self.assertEqual(profit_wwr, 50)
            e = None
            for c in cells:
                if c['l'] == 'cost_num' and c['c'] == 'test_1' and c['p'] == '2017: q1' and c['r'] == 'WWR':
                    e = c['e']
            self.assertIsNotNone(e)
            self.assertFalse(e)

    def test_pop(self):
        rc = hfod.ReportCalc('test_1', 2017, 'yr', 'usd')
        self.assertAlmostEqual(rc.simulate_dynamic_obj({'expr': "=POP('revenue_num', 1)"})['us_obj'], 0.11, places=2)
        self.assertAlmostEqual(rc.simulate_dynamic_obj({'expr': "=POP('revenue_num')"})['us_obj'], 0.11, places=2)

    @unittest.skip("requires sjclient")
    def test_preview_expression(self):
        with self.updated_settings({
            "dynamic_obj": {"expr": "=RO_CF_GROWTH_NUM*2", "g_expr": ""},
            "visibility_status": "Visible To Clients",
            "region": "non_geog",
            "company": "AE",
            "periodicity": "yr",
            "notes": "",
            "dynamic_req_dependencies": None
        }):
            hfod.preview_expression()

    def test_get_relative_year_period(self):
        f = hfod.Func({}, 'US', 'test_1', 2017, 'yr')
        self.assertEqual(f.get_relative_year_period(), (2017, 'yr'), msg='no change')
        self.assertEqual(f.get_relative_year_period(-5), (2012, 'yr'), msg='5 yrs ago')
        self.assertEqual(f.get_relative_year_period(-5, -2), (2010, 'yr'), msg='7 yrs ago')
        self.assertEqual(f.get_relative_year_period(-5, 'h1'), (2012, 'h1'), msg='set h1')
        self.assertEqual(f.get_relative_year_period(2016, 'h1'), (2016, 'h1'), msg='very absolute')
        f = hfod.Func({}, 'US', 'test_1', 2017, 'q2')
        self.assertEqual(f.get_relative_year_period(2016), (2016, 'q2'), msg='no change')
        self.assertEqual(f.get_relative_year_period(2016, -1), (2016, 'q1'), msg='prev quarter')
        self.assertEqual(f.get_relative_year_period(2016, -2), (2015, 'q4'), msg='prev quarter')
        self.assertEqual(f.get_relative_year_period(2016, -3), (2015, 'q3'), msg='prev quarter')
        self.assertEqual(f.get_relative_year_period(2016, -4), (2015, 'q2'), msg='prev quarter')
        self.assertEqual(f.get_relative_year_period(2016, -8), (2014, 'q2'), msg='prev quarter')
        self.assertEqual(f.get_relative_year_period(2016, -18), (2011, 'q4'), msg='prev quarter')
        self.assertEqual(f.get_relative_year_period(2016, 0), (2016, 'q2'), msg='this quarter')
        self.assertEqual(f.get_relative_year_period(2016, 1), (2016, 'q3'), msg='next quarter')
        self.assertEqual(f.get_relative_year_period(2016, 2), (2016, 'q4'), msg='next quarter')
        self.assertEqual(f.get_relative_year_period(2016, 3), (2017, 'q1'), msg='next quarter')

    def test_required_line_item(self):
        rc = hfod.ReportCalc('test_1', 2015, 'yr', 'usd')
        self.assertAlmostEqual(rc.calc_report()['noram_obj']['revenue_over_cost_num'].value,
                               0.98,
                               places=2,
                               msg='canada is excluded...')

    def test_trail(self):
        rc = hfod.ReportCalc('test_1', 2017, 'yr', 'usd')
        self.assertEqual(
            rc.simulate_dynamic_obj({'expr': "=TRAIL('revenue_num', 2)"}),
            {
                'regroll_obj': 150.0,
                'us_obj': 281.0,
                'wwr_obj': 150.0,
                'can_obj': 150.0,
                'con_obj': 150.0,
                'regwest_obj': 150.0,
                'noram_obj': 150.0
            }
        )
        self.assertAlmostEqual(
            rc.simulate_dynamic_obj({'expr': "=TRAIL('revenue_num', 2, 'avg')"})['us_obj'],
            93.67,
            places=2
        )
        self.assertEqual(rc.simulate_dynamic_obj({'expr': "=TRAIL('revenue_num', 2, 'med')"})['us_obj'], 91.0)

        rc = hfod.ReportCalc('test_1', 2017, 'q4', 'usd')
        self.assertEqual(rc.simulate_dynamic_obj({'expr': "=TRAIL('revenue_num', 1)"})['us_obj'], 47.5)

    def test_ttm(self):
        rc = hfod.ReportCalc('test_1', 2017, 'q1', 'usd')
        self.assertEqual(rc.simulate_dynamic_obj({'expr': "=TTM('revenue_num', ifnull=0)"})['wwr_obj'], 100)

    @staticmethod
    @contextlib.contextmanager
    def updated_settings(s):
        original_settings = copy.deepcopy(hfod.settings)
        hfod.settings.update(s)
        yield
        hfod.settings = original_settings


if __name__ == '__main__':
    hfod.get_report = get_report
    hfod.get_dynamic_objs = get_dynamic_objs
    hfod.get_companies = get_companies
    unittest.main()
