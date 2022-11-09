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
    0: 'none',
    1: '1. threats, plans to harm and incitement',
    2: '2. derogation',
    3: '3. animosity',
    4: '4. prejudiced discussions',
}

VEC_LABEL_NUM_TO_STR = {
    0: 'none',
    1: '1.1 threats of harm',
    2: '1.2 incitement and encouragement of harm',
    3: '2.1 descriptive attacks',
    4: '2.2 aggressive and emotive attacks',
    5: '2.3 dehumanising attacks & overt sexual objectification',
    6: '3.1 casual use of gendered slurs, profanities, and insults',
    7: '3.2 immutable gender differences and gender stereotypes',
    8: '3.3 backhanded gendered compliments',
    9: '3.4 condescending explanations or unwelcome advice',
    10: '4.1 supporting mistreatment of individual women',
    11: '4.2 supporting systemic discrimination against women as a group',
}

LABEL_STR_TO_LABEL_NUM = {
    # task A
    'not sexist': 0,
    'sexist': 1,
    # task B
    'none': 0,
    '1. threats, plans to harm and incitement': 1,
    '2. derogation': 2,
    '3. animosity': 3,
    '4. prejudiced discussions': 4,
    # task C
    '1.1 threats of harm': 1,
    '1.2 incitement and encouragement of harm': 2,
    '2.1 descriptive attacks': 3,
    '2.2 aggressive and emotive attacks': 4,
    '2.3 dehumanising attacks & overt sexual objectification': 5,
    '3.1 casual use of gendered slurs, profanities, and insults': 6,
    '3.2 immutable gender differences and gender stereotypes': 7,
    '3.3 backhanded gendered compliments': 8,
    '3.4 condescending explanations or unwelcome advice': 9,
    '4.1 supporting mistreatment of individual women': 10,
    '4.2 supporting systemic discrimination against women as a group': 11,
}