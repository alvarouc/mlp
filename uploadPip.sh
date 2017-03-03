#!/bin/bash

# First change manually the version
version="$(python setup.py --version)"
echo "Current version $version"
read -p "Enter new version:"  newVersion
sed -i "s/$version/$newVersion/g" setup.py
git tag "$newVersion" -m "from $version to $newVersion"
git push --tags origin master

python setup.py register -r pypi
python setup.py sdist upload -r pypi
