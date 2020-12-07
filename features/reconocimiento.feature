Feature: Model verification

  Scenario Outline: Verify mask model
    Given an <image> and service with the <url>
    When the user wants to validate the mask usage
    Then the model gives a <result>

    Examples:
      | url                                   | image           | result        |
      | http://127.0.0.1:5000/api/recognition | image_pivot.jpg | Con tapabocas |
      | http://127.0.0.1:5000/api/recognition | image_pivot.jpg | Sin tapabocas |