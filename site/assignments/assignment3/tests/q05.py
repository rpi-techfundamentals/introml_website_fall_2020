test = {
  'name': 'Question',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> train['family'].sum()
          506
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> test['family'].sum()
          186
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
