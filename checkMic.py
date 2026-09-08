from record import list_input_devices


for device in list_input_devices():
    print(
        f"{device.index}: {device.name} | "
        f"input channels={device.max_input_channels} | "
        f"default rate={device.default_sample_rate}"
    )
