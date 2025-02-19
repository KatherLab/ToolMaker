def pytest_addoption(parser):
    parser.addoption(
        "--prefix",
        action="store",
        default=None,
        help="Prefix for the tool names, e.g. 'toolmaker' or 'openhands'",
    )


def pytest_collection_modifyitems(session, config, items):
    for item in items:
        for marker in item.iter_markers(name="tool_invocation"):
            item.user_properties.append(("prefix", marker.kwargs["prefix"]))
            item.user_properties.append(("tool", marker.kwargs["tool"]))
            item.user_properties.append(("invocation", marker.kwargs["invocation"]))
