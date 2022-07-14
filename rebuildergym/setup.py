from setuptools import setup

setup(
    name='rebuildergym',
    version='0.1.0',
    keywords='games, environment, agent, rl, ai, gym',
    description='Rebuilder Gym Environment',
    packages=['rebuildergym'],
    install_requires=[
        'gym>=0.9.4',
        'numpy>=1.13.0',
        'opencv-python>=3.4.2.0'
    ]
)
