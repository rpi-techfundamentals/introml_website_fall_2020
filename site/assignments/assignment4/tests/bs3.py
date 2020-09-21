test = {
  'name': 'Question',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> json_links[10][-7:]
          'json.gz'
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> csv_links[10][-4:]
          '.zip'
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> json_links[9][:62]
          'https://s3.amazonaws.com/weruns/forfun/Kickstarter/Kickstarter'
          """,
          'hidden': False,
          'locked': False
        },
        {
          'code': r"""
          >>> json_links[9][:62]
          'https://s3.amazonaws.com/weruns/forfun/Kickstarter/Kickstarter'
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
