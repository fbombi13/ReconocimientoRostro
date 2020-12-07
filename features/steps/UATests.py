import requests
from behave import *

use_step_matcher("re")


@given("an (?P<image>.+) and service with the (?P<url>.+)")
def step_impl(context, image, url):
    """
    :type context: behave.runner.Context
    :type image: str
    :type url: str
    """
    context.image_path = image
    context.url = url


@when("the user wants to validate the mask usage")
def step_impl(context):
    """
    :type context: behave.runner.Context
    """
    params = {'image': 'image_pivot.jpg'}
    response = requests.post(context.url, params=params)
    context.response = response.text


@then("the model gives a (?P<result>.+)")
def step_impl(context, result):
    """
    :type context: behave.runner.Context
    :type result: str
    """
    assert context.response == result
