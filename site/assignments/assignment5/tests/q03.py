test = {
  'name': 'Question',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> int(knn0_train_y.sum())==241
          True
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> int(knn0_val_y.sum())==69
          True
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> round(knn0_train_accuracy,2)==0.81
          True
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> round(knn0_val_accuracy,2)==0.68
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
