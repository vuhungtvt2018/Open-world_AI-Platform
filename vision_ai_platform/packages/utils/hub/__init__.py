# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from vision_ai_platform.packages.utils.hub.auth import Auth
from vision_ai_platform.packages.utils.hub.session import HUBTrainingSession
from vision_ai_platform.packages.utils.hub.utils import HUB_API_ROOT, HUB_WEB_ROOT, PREFIX
from vision_ai_platform.packages.utils import LOGGER, SETTINGS, check

__all__ = (
    "HUB_WEB_ROOT",
    "PREFIX",
    "HUBTrainingSession",
    "check_dataset",
    "export_fmts_hub",
    "export_model",
    "get_export",
    "login",
    "logout",
    "reset_model",
)


def login(api_key: str | None = None, save: bool = True) -> bool:
    """Log in to the Ultralytics HUB API using the provided API key.

    The session is not stored; a new session is created when needed using the saved SETTINGS if successfully
    authenticated.

    Args:
        api_key (str, optional): API key to use for authentication. If not provided, it will be retrieved from SETTINGS.
        save (bool, optional): Whether to save the API key to SETTINGS if authentication is successful.

    Returns:
        (bool): True if authentication is successful, False otherwise.
    """
    check.check_requirements("hub-sdk>=0.0.12")
    from hub_sdk import HUBClient

    api_key_url = f"{HUB_WEB_ROOT}/settings?tab=api+keys"  # set the redirect URL
    saved_key = SETTINGS.get("api_key")
    active_key = api_key or saved_key
    credentials = {"api_key": active_key} if active_key and active_key != "" else None  # set credentials

    client = HUBClient(credentials)  # initialize HUBClient

    if client.authenticated:
        # Successfully authenticated with HUB

        if save and client.api_key != saved_key:
            SETTINGS.update({"api_key": client.api_key})  # update settings with valid API key

        # Set message based on whether key was provided or retrieved from settings
        log_message = (
            "New authentication successful ✅" if client.api_key == api_key or not credentials else "Authenticated ✅"
        )
        LOGGER.info(f"{PREFIX}{log_message}")

        return True
    else:
        # Failed to authenticate with HUB
        LOGGER.info(f"{PREFIX}Get API key from {api_key_url} and then run 'yolo login API_KEY'")
        return False


def logout():
    """Log out of Ultralytics HUB by removing the API key from the settings file."""
    SETTINGS["api_key"] = ""
    LOGGER.info(f"{PREFIX}logged out ✅. To log in again, use 'yolo login'.")


def reset_model(model_id: str = ""):
    """Reset a trained model to an untrained state."""
    import requests  # scoped as slow import

    r = requests.post(
        f"{HUB_API_ROOT}/model-reset",
        json={"modelId": model_id},
        headers={"x-api-key": Auth().api_key},
        timeout=30,
    )
    if r.status_code == 200:
        LOGGER.info(f"{PREFIX}Model reset successfully")
        return
    LOGGER.warning(f"{PREFIX}Model reset failure {r.status_code} {r.reason}")


def export_fmts_hub():
    """Return a list of HUB-supported export formats."""
    from vision_ai_platform.packages.core.exporter import export_formats

    return [*list(export_formats()["Argument"][1:]), "ultralytics_tflite", "ultralytics_coreml"]


def export_model(model_id: str = "", format: str = "torchscript"):
    """Export a model to a specified format for deployment via the Ultralytics HUB API.

    Args:
        model_id (str): The ID of the model to export. An empty string will use the default model.
        format (str): The format to export the model to. Must be one of the supported formats returned by
            export_fmts_hub().

    Raises:
        AssertionError: If the specified format is not supported or if the export request fails.

    Examples:
        >>> from ultralytics import hub
        >>> hub.export_model(model_id="your_model_id", format="torchscript")
    """
    import requests  # scoped as slow import

    assert format in export_fmts_hub(), f"Unsupported export format '{format}', valid formats are {export_fmts_hub()}"
    r = requests.post(
        f"{HUB_API_ROOT}/v1/models/{model_id}/export",
        json={"format": format},
        headers={"x-api-key": Auth().api_key},
        timeout=30,
    )
    assert r.status_code == 200, f"{PREFIX}{format} export failure {r.status_code} {r.reason}"
    LOGGER.info(f"{PREFIX}{format} export started ✅")


def get_export(model_id: str = "", format: str = "torchscript"):
    """Retrieve an exported model in the specified format from Ultralytics HUB using the model ID.

    Args:
        model_id (str): The ID of the model to retrieve from Ultralytics HUB.
        format (str): The export format to retrieve. Must be one of the supported formats returned by export_fmts_hub().

    Returns:
        (dict): JSON response containing the exported model information.

    Raises:
        AssertionError: If the specified format is not supported or if the API request fails.

    Examples:
        >>> from ultralytics import hub
        >>> result = hub.get_export(model_id="your_model_id", format="torchscript")
    """
    import requests  # scoped as slow import

    assert format in export_fmts_hub(), f"Unsupported export format '{format}', valid formats are {export_fmts_hub()}"
    r = requests.post(
        f"{HUB_API_ROOT}/get-export",
        json={"apiKey": Auth().api_key, "modelId": model_id, "format": format},
        headers={"x-api-key": Auth().api_key},
        timeout=30,
    )
    assert r.status_code == 200, f"{PREFIX}{format} get_export failure {r.status_code} {r.reason}"
    return r.json()
