import chainlit as cl

@cl.action_callback("action_button")
async def on_action(action):
    await cl.Message(content=f"Executed {action.name}").send()
    await action.remove()
    await cl.Message(content=f"Đã mở phòng thử đồ {action.name}").send()

@cl.action_callback("inter_virtual_fittingroom")
async def on_action(action):
    await cl.Message(content=f"Executed {action.name}").send()
    await action.remove()

@cl.on_chat_start
async def start():
    # Sending an action button within a chatbot message
    actions = [
        cl.Action(name="action_button", value="example_value", description="Click me!"),
        cl.Action(name="inter_virtual_fittingroom", value="example_value", description="Click me!")
    ]

    await cl.Message(content="Interact with this action button:", actions=actions).send()