def wan_video_vae_state_dict_converter(state_dict):
    converted = {}
    if "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    for name, value in state_dict.items():
        converted[f"model.{name}"] = value
    return converted
