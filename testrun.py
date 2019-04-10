import hfod_calc as hfod

TEST_REPORTS = {
    'test_1.2017.yr.usd': {
        'us_obj': {
            'profit_num':94.0,
            'revenue_num':100.0,
            'cost_num':6,
        },
        'can_obj': {
            'revenue_num':50.0,
            'cost_num':6,
        },
        'wwf_obj':{
            'gi_fiscalpe_num': 10
        }
    },
    'test_2.2017.yr.usd': {
        'us_obj': {
            'revenue_num':120.0,
            'cost_num':5,
        },
        'can_obj': {
            'revenue_num':50.0,
            'cost_num':7,
        }
    },
    'test_3.2017.yr.usd': {
        'noram_obj': {
            'revenue_num':50.0,
            'cost_num':50,
        }
    },
    'test_1.2016.yr.usd': {
        'us_obj': {
            'revenue_num':90.0,
            'cost_num':89.0,
        },
        'can_obj': {
            'revenue_num':50.0,
            'cost_num':6,
        },
        'wwf_obj':{
            'gi_fiscalpe_num': 10
        }
    },
    'test_1.2015.yr.usd': {
        'us_obj': {
            'revenue_num':91.0,
            'cost_num':93.0,
        },
        'can_obj': {
            'revenue_num':50.0,
        }
    },
    'test_1.2017.q1.usd': {
        'us_obj': {
            'revenue_num':100.0,
            'cost_num':22.5,
            'profit_num':77.5
        }
    },
    'test_1.2017.q4.usd': {
        'us_obj': {
            'revenue_num':25,
            'cost_num':22.5,
        },
        'wwf_obj':{
            'gi_fiscalpe_num': 11
        }
    },
    'test_1.2016.q4.usd': {
        'us_obj': {
            'revenue_num':22.5,
            'cost_num':22.5,
        }
    }
}

def get_report(company, year, period, currency='usd'):
    key = u'{}.{}.{}.{}'.format(company, year, period, currency)
    return TEST_REPORTS.get(key, {})

def get_dynamic_objs(period):
    return [
        {
            'field': 'double_double_profit_num',
            'expr':'=double_profit_num*2',
            'dd': ['double_profit_num']
        },
        {
            'field': 'double_profit_num',
            'expr':'=profit_num*2',
            'dd': ['profit_num']
        },
        {
            'field': 'revenue_over_cost_num',
            'expr':'=revenue_num/cost_num',
            'req': ['revenue_num','cost_num']
        },
        {
            'field':'profit_num',
            'expr':'=revenue_num-cost_num',
        }
    ]

def get_companies():
    return [{
        'id':'test_1',
        'peer_groups':[
            'group_a',
            'group_b'
        ]
    }, {
        'id':'test_2',
        'peer_groups':[
            'group_a',
            'group_c'
        ]
    }, {
        'id':'test_3',
        'peer_groups':[
            'group_a',
            'group_b'
        ]
    }]


hfod.get_report = get_report
hfod.get_dynamic_objs = get_dynamic_objs
hfod.get_companies = get_companies
hfod.test()