test = {
  'name': 'Question',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> generate_submission(train, 'PredEveryoneDies', 'submiteveryonedies.csv').columns[1]
          'Survived'
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> generate_submission(train, 'PredEveryoneDies', 'submiteveryonedies.csv').shape
          (891, 2)
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
