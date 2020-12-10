test = {
  'name': 'Question',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> int(train_y.sum())==267
          True
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> int(val_y.sum())==75
          True
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> int(train_X.Cabin_H.sum())==539
          True
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> int(val_X.Cabin_H.sum())==149
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
