test = {
  'name': 'Question',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> list1
          [5, 7, 9, 11, 13, 15, 17]
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> list_step(5, 19, 2)
          [5, 7, 9, 11, 13, 15, 17]
          """,
          'hidden': False,
          'locked': False
        },
               {
          'code': r"""
          >>> list_step(5, 56, 7)
          [5, 12, 19, 26, 33, 40, 47, 54]
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
