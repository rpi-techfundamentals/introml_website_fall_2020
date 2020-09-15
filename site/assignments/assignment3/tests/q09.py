test = {
  'name': 'Question',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> int(sum(train.PredGender))
          314
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> int(sum(test.PredGender))
          152
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> round(AccGender,2)
          78.68
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
