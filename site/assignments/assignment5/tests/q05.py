test = {
  'name': 'Question',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> int(knn0_con_train[0,0])==389
          True
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> int(knn0_con_train[0,1])==56
          True
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> int(knn0_con_val[1,0])==32
          True
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> int(knn0_con_val[1,1])==43
          True
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
