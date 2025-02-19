import random
import string

from toolmaker.actions.actions import Observation, truncate_observation


def random_string(length: int) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def test_truncate_observation_shorter_than_max_length():
    observation = Observation(content="Hello, world!")
    truncated_observation = truncate_observation(observation, max_length=100)
    assert truncated_observation is observation


def test_truncate_observation_longer_than_max_length():
    observation = Observation(content=random_string(1000))
    truncated_observation = truncate_observation(observation, max_length=100)
    assert observation.content not in truncated_observation.content
    assert observation.content[:50] in truncated_observation.content
    assert observation.content[-50:] in truncated_observation.content
    assert "<truncated/>" in truncated_observation.content
    assert "The content had to be truncated because its length"
