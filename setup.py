from setuptools import setup, find_packages


def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    git_requirements = [l for l in lines if l.startswith('git+')]
    requirements = [l for l in lines if not l.startswith('git+')]
    return requirements, git_requirements


install_requires, dependency_links = parse_requirements('requirements.txt')

setup(
    name='dfd-arena',
    version='0.1.0',
    packages=find_packages(include=['arena', 'arena.*']),
    install_requires=install_requires,
    dependency_links=dependency_links
)
