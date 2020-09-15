test = {
  'name': 'Question',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> sum(train.Age.isna())
          0
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> sum(test.Age.isna())
          0
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> round(train.Age.mean(),2)
          29.36
          """,
          'hidden': False,
          'locked': False
        },        
        {
          'code': r"""
          >>> round(test.Age.mean(),2)
          29.6
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
