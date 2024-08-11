# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from dataclasses import dataclass
from pwd import getpwuid
from typing import List, Union

from download import is_model_downloaded, load_model_configs

"""Helper functions for the OpenAI API Models endpoint.

See https://platform.openai.com/docs/api-reference/models for the full specification and details.
Please create an issue if anything doesn't match the specification.
"""


@dataclass
class ModelInfo:
    """The Model object per the OpenAI API specification containing information about a model.

    See https://platform.openai.com/docs/api-reference/models/object for more details.
    """

    id: str
    created: int
    owned_by: str
    object: str = "model"


@dataclass
class ModelInfoList:
    """A list of ModelInfo objects."""

    data: List[ModelInfo]
    object: str = "list"


def retrieve_model_info(args, model_id: str) -> Union[ModelInfo, None]:
    """Implementation of the OpenAI API Retrieve Model endpoint.

    See https://platform.openai.com/docs/api-reference/models/retrieve

    Inputs:
        args: command line arguments
        model_id: the id of the model requested

    Returns:
        ModelInfo describing the specified if it is downloaded, None otherwise.
    """
    if model_config := load_model_configs().get(model_id):
        if is_model_downloaded(model_id, args.model_directory):
            path = args.model_directory / model_config.name
            created = int(os.path.getctime(path))
            owned_by = getpwuid(os.stat(path).st_uid).pw_name

            return ModelInfo(id=model_config.name, created=created, owned_by=owned_by)
        return None
    return None


def get_model_info_list(args) -> ModelInfo:
    """Implementation of the OpenAI API List Models endpoint.

    See https://platform.openai.com/docs/api-reference/models/list

    Inputs:
        args: command line arguments

    Returns:
        ModelInfoList describing all downloaded models.
    """
    data = []
    for model_id, model_config in load_model_configs().items():
        if is_model_downloaded(model_id, args.model_directory):
            path = args.model_directory / model_config.name
            created = int(os.path.getctime(path))
            owned_by = getpwuid(os.stat(path).st_uid).pw_name

            data.append(
                ModelInfo(id=model_config.name, created=created, owned_by=owned_by)
            )
    response = ModelInfoList(data=data)
    return response
