test = {
  'name': 'Question',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> len(train["PredEveryoneDies"])
          891
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> len(test["PredEveryoneDies"])
          418
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
