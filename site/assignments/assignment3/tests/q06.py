test = {
  'name': 'Question',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> generate_accuracy(df.predicted, df.actual)
          50.0
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> round(generate_accuracy(df.loc[0:2,'predicted'], df.loc[0:2,'actual']),2)
          33.33
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
