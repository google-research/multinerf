from setuptools import setup

# reading long description from file
with open('README.md') as file:
        long_description = file.read()


# specify requirements of your package here
with open('requirements.txt') as file:
        REQUIREMENTS = file.readlines()

# some more details
CLASSIFIERS = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Researchers',
        'Topic :: Internet',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        ]

# calling the setup function
setup(name='multinerf',
        version='1.0.0',
        description='this is fork of https://github.com/google-research/multinerf.git for pypi',
        long_description=long_description,
        url='https://github.com/google-research/multinerf.git',
        author='jonbarron',
        author_email='jonbarron@gmail.com',
        license='MIT',
        classifiers=CLASSIFIERS,
        install_requires=REQUIREMENTS,
        keywords='maps location address'
        )
