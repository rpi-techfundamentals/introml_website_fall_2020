# Jupyter Class

This is a project to easily build a course website from Jupyter Book.  

Set the configuration in the config folder.

## Building a Jupyter Book

Run the following command in your terminal:

```bash
jb build site/
```

If you would like to work with a clean build, you can empty the build folder by running:

```bash
jb clean site/
```

If jupyter execution is cached, this command will not delete the cached folder.

To remove the build folder (including `cached` executables), you can run:

```bash
jb clean --all site/
```

## Publishing this Jupyter Book

This repository is published automatically to `gh-pages` upon `push` to the `master` branch.

## Notes

This repository is used as a test case for [jupyter-book](https://github.com/executablebooks/jupyter-book) and
a `requirements.txt` file is provided to support this `CI` application.
