import os
import json
import requests
import streamlit as st
from google import genai
from google.genai import types
import aiofiles

# Set environment variable for Google credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './sqy-prod.json'

async def process_image(client, image_data: bytes):
    # Save uploaded file temporarily
    temp_file = "temp_image.jpeg"
    async with aiofiles.open(temp_file, 'wb') as f:
        await f.write(image_data)
    uploaded_file = client.files.upload(file=temp_file)
    file_uri = uploaded_file.uri
    mime_type = uploaded_file.mime_type
    # Clean up temporary file
    os.remove(temp_file) 

    model = "gemini-2.0-flash-lite"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=file_uri,
                    mime_type=mime_type,
                ),
                types.Part.from_text(text="""Classify this image"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            properties={
                "classification": genai.types.Schema(
                    type=genai.types.Type.STRING,
                    enum=[
                          'Bathroom', 'Bedroom', 'Living Room', 'Exterior View', 'Kitchen', 'Garden', 'Plot Area', 'Room',
                'Swimming Pool', 'Gym', 'Parking', 'Map Location', 'Balcony', 'Floor Plan', 'Furnished Amenities',
                'Building Lobby', 'Team Area', 'Staircase', 'Master Plan','false'
                    ],
                ),
            },
        ),
        system_instruction=[
            types.Part.from_text(text="""
                You are an image classifier for images shown in real estate listings.
                - If image not related to real-estate then return 'false'.
                """),
        ],
    )
    result = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        result += chunk.text
    
    # Parse the JSON response and extract just the classification value
    parsed_result = json.loads(result)
    return parsed_result["classification"]

async def main():
    st.title("Image Classification App")
    
    # Initialize the Google AI client
    client = genai.Client(api_key="AIzaSyBeNEyjpgf8tX2AQJSunqSIOBfPGr08DS8")
    
    # Two tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Image", "Image URL"])
    
    # Tab 1: File Upload
    with tab1:
        uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            try:
                # Read image data
                image_data = uploaded_file.read()
                # Display the uploaded image
                st.image(image_data, caption="Uploaded Image", use_column_width=True)
                
                if st.button("Classify Uploaded Image"):
                    with st.spinner("Classifying image..."):
                        classification = await process_image(client, image_data)
                        st.success(f"Classification result: {classification}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Tab 2: URL Input
    with tab2:
        image_url = st.text_input("Enter image URL")
        if image_url:  # This will trigger automatically when URL is entered
            try:
                # Download and display the image automatically
                response = requests.get(image_url)
                response.raise_for_status()
                image_data = response.content
                st.image(image_data, caption="Image from URL", use_column_width=True)
                
                # Separate button for classification
                if st.button("Classify URL Image"):
                    with st.spinner("Classifying image..."):
                        classification = await process_image(client, image_data)
                        st.success(f"Classification result: {classification}")
            except requests.RequestException as e:
                st.error(f"Failed to download image from URL: {str(e)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())