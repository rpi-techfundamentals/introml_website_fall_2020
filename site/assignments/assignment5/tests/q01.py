test = {
  'name': 'Question',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> X_train.columns.values.tolist()== ['Age', 'SibSp', 'Parch', 'Fare', 'family_size', 'Pclass_2', 'Pclass_3', 'Sex_male', 'Cabin_B', 'Cabin_C', 'Cabin_D', 'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_H', 'Embarked_Q', 'Embarked_S']
          True
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> X_test.columns.values.tolist()== ['Age', 'SibSp', 'Parch', 'Fare', 'family_size', 'Pclass_2', 'Pclass_3', 'Sex_male', 'Cabin_B', 'Cabin_C', 'Cabin_D', 'Cabin_E', 'Cabin_F', 'Cabin_G', 'Cabin_H', 'Embarked_Q', 'Embarked_S']
          True
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> int(y.sum())== 342
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
