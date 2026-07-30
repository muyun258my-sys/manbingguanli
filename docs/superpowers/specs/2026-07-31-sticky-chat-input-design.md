# Sticky Chat Input Design

## Goal

Rework the Streamlit chat panel so its question input remains fixed at the top of the chat column while conversation history is rendered below it from oldest to newest.

## Layout

The chat column has two distinct areas:

1. A sticky input form at the top, containing a single-line question field and submit button.
2. A chronological history region below it, rendered exclusively from `st.session_state.messages`.

The health-profile panel remains in the right-hand column.

## Interaction

Submitting a question appends the user message first. The UI then requests `/chat`, appends a complete assistant message on success, and reruns so both messages appear in their stable chronological positions. Request errors append an assistant error message, preserving the conversational sequence.

While a request is in progress, the input is disabled and the submit button cannot issue a second request. Enter submits the Streamlit form.

## Styling

A scoped CSS rule makes the input wrapper `position: sticky` with `top: 0`, a matching page background, and stacking order above history content. The history region remains ordinary document flow, so it scrolls underneath the fixed input area without overlap.

## Validation

Check Python syntax, start Streamlit, and inspect desktop and narrow-screen views. Verify the input is at the top and remains visible while scrolling, messages are old-to-new, submitting creates one user message and one assistant message in that order, and the profile panel remains usable.
