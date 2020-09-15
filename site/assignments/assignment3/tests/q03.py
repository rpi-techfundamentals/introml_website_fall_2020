test = {
  'name': 'Question',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> list3
          [4, 8, 16, 20, 28, 32, 40, 44, 52, 56, 64, 68, 76, 80, 88, 92]
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> list_divisible_not(300, 500, 11, 3)
          [308, 319, 341, 352, 374, 385, 407, 418, 440, 451, 473, 484]
          """,
          'hidden': False,
          'locked': False
        },
               {
          'code': r"""
          >>> list_divisible_not(200, 300, 7, 8)
          [203, 210, 217, 231, 238, 245, 252, 259, 266, 273, 287, 294]
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