from setuptools import setup, find_packages

# Read the contents of the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='semiconv',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    description='...',
    author='William Duckett',
    author_email='will.duckett@warwick.ac.uk',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='==3.9.6',
)