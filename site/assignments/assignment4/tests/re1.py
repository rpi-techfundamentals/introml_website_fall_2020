test = {
  'name': 'Question',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> blind_text(my_text)[0]
          'the test score is ** with the email -- for mgmt*****.'
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> len(blind_text(my_text))
          2
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
