VECTOR_TO_BIN_MAPPING = {
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 1,
    10: 1,
    11: 1
}

VECTOR_TO_CAT_MAPPING = {
    0: 0,
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 2,
    6: 3,
    7: 3,
    8: 3,
    9: 3,
    10: 4,
    11: 4
}

BIN_LABEL_NUM_TO_STR = {
    0: 'not sexist',
    1: 'sexist'
}

CAT_LABEL_NUM_TO_STR = {
    # 0: 'none',
    0: '1. threats, plans to harm and incitement',
    1: '2. derogation',
    2: '3. animosity',
    3: '4. prejudiced discussions',
}

VEC_LABEL_NUM_TO_STR = {
    # 0: 'none',
    0: '1.1 threats of harm',
    1: '1.2 incitement and encouragement of harm',
    2: '2.1 descriptive attacks',
    3: '2.2 aggressive and emotive attacks',
    4: '2.3 dehumanising attacks & overt sexual objectification',
    5: '3.1 casual use of gendered slurs, profanities, and insults',
    6: '3.2 immutable gender differences and gender stereotypes',
    7: '3.3 backhanded gendered compliments',
    8: '3.4 condescending explanations or unwelcome advice',
    9: '4.1 supporting mistreatment of individual women',
    10: '4.2 supporting systemic discrimination against women as a group',
}

LABEL_STR_TO_LABEL_NUM = {
    # task A
    'not sexist': 0,
    'sexist': 1,
    # task B
    # 'none': 0,
    '1. threats, plans to harm and incitement': 0,
    '2. derogation': 1,
    '3. animosity': 2,
    '4. prejudiced discussions': 3,
    # task C
    '1.1 threats of harm': 0,
    '1.2 incitement and encouragement of harm': 1,
    '2.1 descriptive attacks': 2,
    '2.2 aggressive and emotive attacks': 3,
    '2.3 dehumanising attacks & overt sexual objectification': 4,
    '3.1 casual use of gendered slurs, profanities, and insults': 5,
    '3.2 immutable gender differences and gender stereotypes': 6,
    '3.3 backhanded gendered compliments': 7,
    '3.4 condescending explanations or unwelcome advice': 8,
    '4.1 supporting mistreatment of individual women': 9,
    '4.2 supporting systemic discrimination against women as a group': 10,
}

# {source: {label_type: {label_value: label_description}}}
GLOBAL_LABEL_MAPPING = {  
    'EDOS2023TaskA': {
        'task_A': {
            1: 'sexist'
        }
    },
    'EDOS2023TaskB': {
        'task_B': {
            0: 'threats, plans to harm and incitement',
            1: 'derogation',
            2: 'animosity',
            3: 'prejudiced discussions'
        }
    },
    'EDOS2023TaskC': {
        'task_C': {
            0: 'threats of harm',
            1: 'incitement and encouragement of harm',
            2: 'descriptive attacks',
            3: 'aggressive and emotive attacks',
            4: 'dehumanising attacks & overt sexual objectification',
            5: 'casual use of gendered slurs, profanities, and insults',
            6: 'immutable gender differences and gender stereotypes',
            7: 'backhanded gendered compliments',
            8: 'condescending explanations or unwelcome advice',
            9: 'supporting mistreatment of individual women',
            10: 'supporting systemic discrimination against women as a group'
        }
    },
    'DGHSD': {
        'hate speech': {
            1: 'hate speech'
        }
    },
    'SBF': {
        'offensive': {
            1: 'offensive'
        },
        'lewd': {
            1: 'lewd'
        }
    },
    'MHS': {
        'targets gender': {
            1: 'targets gender'
        },
        'targets women': {
            1: 'targets women'
        }
    },
    'TWE': {
        'hate': {
            1: 'hate'
        },
        'irony': {
            1: 'irony'
        },
        'sentiment': {
            0: 'sentiment: negative',
            1: 'sentiment: neutral',
            2: 'sentiment: positive'
        },
        'stance_abortion': {
            0: 'stance abortion: none',
            1: 'stance abortion: against',
            2: 'stance abortion: favor'
        },
        'stance_feminist': {
            0: 'stance feminist: none',
            1: 'stance feminist: against',
            2: 'stance feminist: favor'
        },
        'emotion': {
            0: 'emotion: anger',
            1: 'emotion: joi',
            2: 'emotion: optimism',
            3: 'emotion: sadness',
        }
    }
}