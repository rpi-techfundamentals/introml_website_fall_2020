test = {
  'name': 'Question',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> int(sum(train.PredGenderAge13))
          351
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> int(sum(test.PredGenderAge13))
          165
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> int(sum(train.PredGenderAge18))
          372
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> int(sum(test.PredGenderAge18))
          176
          """,
          'hidden': False,
          'locked': False
        }
      ],
      'scored': True,
      'setup': '',
      'teardown': '',
      'type': 'doctest'
    }
  ]
}
