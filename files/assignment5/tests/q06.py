test = {
  'name': 'Question',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> round(knn1_train_accuracy,2)>0.81
          True
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> round(knn1_val_accuracy,2)>0.68
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
