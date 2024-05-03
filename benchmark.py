import argparse
from typing import List

import ollama
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

from datetime import datetime


class Message(BaseModel):
    role: str
    content: str


class OllamaResponse(BaseModel):
    model: str
    created_at: datetime
    message: Message
    done: bool
    total_duration: int
    load_duration: int = 0
    prompt_eval_count: int = Field(-1, validate_default=True)
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int

    @field_validator("prompt_eval_count")
    @classmethod
    def validate_prompt_eval_count(cls, value: int) -> int:
        if value == -1:
            print(
                "\nWarning: prompt token count was not provided, potentially due to prompt caching. For more info, see https://github.com/ollama/ollama/issues/2068\n"
            )
            return 0  # Set default value
        return value


def run_benchmark(
    model_name: str, prompt: str, verbose: bool
) -> OllamaResponse:

    last_element = None

    if verbose:
        stream = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            stream=True,
        )
        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
            last_element = chunk
    else:
        last_element = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

    if not last_element:
        print("System Error: No response received from ollama")
        return None

    # with open("data/ollama/ollama_res.json", "w") as outfile:
    #     outfile.write(json.dumps(last_element, indent=4))

    return OllamaResponse.model_validate(last_element)


def nanosec_to_sec(nanosec):
    return nanosec / 1000000000


def inference_stats(model_response: OllamaResponse):
    # Use properties for calculations
    prompt_ts = model_response.prompt_eval_count / (
        nanosec_to_sec(model_response.prompt_eval_duration)
    )
    response_ts = model_response.eval_count / (
        nanosec_to_sec(model_response.eval_duration)
    )
    total_ts = (
        model_response.prompt_eval_count + model_response.eval_count
    ) / (
        nanosec_to_sec(
            model_response.prompt_eval_duration + model_response.eval_duration
        )
    )

    print(
        f"""
----------------------------------------------------
        {model_response.model}
        \tPrompt eval: {prompt_ts:.2f} t/s
        \tResponse: {response_ts:.2f} t/s
        \tTotal: {total_ts:.2f} t/s

        Stats:
        \tPrompt tokens: {model_response.prompt_eval_count}
        \tResponse tokens: {model_response.eval_count}
        \tModel load time: {nanosec_to_sec(model_response.load_duration):.2f}s
        \tPrompt eval time: {nanosec_to_sec(model_response.prompt_eval_duration):.2f}s
        \tResponse time: {nanosec_to_sec(model_response.eval_duration):.2f}s
        \tTotal time: {nanosec_to_sec(model_response.total_duration):.2f}s
----------------------------------------------------
        """
    )


def average_stats(responses: List[OllamaResponse]):
    if len(responses) == 0:
        print("No stats to average")
        return

    res = OllamaResponse(
        model=responses[0].model,
        created_at=datetime.now(),
        message=Message(
            role="system",
            content=f"Average stats across {len(responses)} runs",
        ),
        done=True,
        total_duration=sum(r.total_duration for r in responses),
        load_duration=sum(r.load_duration for r in responses),
        prompt_eval_count=sum(r.prompt_eval_count for r in responses),
        prompt_eval_duration=sum(r.prompt_eval_duration for r in responses),
        eval_count=sum(r.eval_count for r in responses),
        eval_duration=sum(r.eval_duration for r in responses),
    )
    print("Average stats:")
    inference_stats(res)


def get_benchmark_models(skip_models: List[str] = []) -> List[str]:
    models = ollama.list().get("models", [])
    model_names = [model["name"] for model in models]
    if len(skip_models) > 0:
        model_names = [
            model for model in model_names if model not in skip_models
        ]
    print(f"Evaluating models: {model_names}\n")
    return model_names


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks on your Ollama models."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase output verbosity",
        default=False,
    )
    parser.add_argument(
        "-s",
        "--skip-models",
        nargs="*",
        default=[],
        help="List of model names to skip. Separate multiple models with spaces.",
    )
    parser.add_argument(
        "-p",
        "--prompts",
        nargs="*",
        default=['Can you take a photo using the back camera and save it to the default location?', 'How can I take a selfie with the front camera?', 'Is it possible to capture a selfie with the front camera?', 'For a quick profile picture, can I take a photo with the front camera and get the image path?', 'Can you help me take a photo of the lecture notes using the back camera?', 'For my blog, I require a photo of myself. Could you use the front camera for a selfie?', 'Document our project progress. Capture a photo with the back camera.', 'I want to capture the night sky. Use the back camera and show me the file path after.', 'Can you show me how to take a selfie using the front camera?', 'Can you show me how to take a selfie using the front camera?', "Remind me to check on the cake in the oven at 17:15. Please set it with the label 'Cake Check'.", "Can you please set a timer alarm for 15:00 with the label 'Nap Time'?", "Can you set an alarm for 09:00 with the label 'Yoga Session'? I want to start my day with some stretching.", "Please set up a timer for 07:15 labeled 'Wake Up Call'.", "How to set a timer alarm at 09:00 with the label 'Project Kickoff' for reminding me of the project starting time?", "Schedule a reminder for 17:30 with the label 'Check Emails'.", "Please, set a timer alarm for 13:45 named 'Coffee Break' to remind me to take a break.", "I need a small nap. Set a timer alarm for 14:20 and call it 'Power Nap', please.", "Please set a timer alarm for 08:00 with the label 'Daily Meditation' to start my day with mindfulness.", "Can you set an alarm for 17:00 with the label 'Workout Time'?", 'Is it possible to adjust my screen to the lowest brightness, level 0, to save battery?', "How can I increment my screen's brightness to level 6 for working on documents?", 'What command would I use to set my screen brightness to maximum for outdoor visibility?', "What's the procedure for setting the screen brightness to a comfortably high level of 8 during daytime?", 'Adjust the brightness to level 5, which is a comfortable setting for my eyes.', 'Is setting the brightness to level 4 good enough for reading without causing eye fatigue?', "Is there a way to adjust my laptop's screen brightness to level 5 for optimal night time use?", 'I need the lowest brightness setting to conserve energy while my device charges.', 'I enjoy a slightly brighter screen in the morning, can you set it to level 6?', "What's the command to change the screen brightness to 5, a medium level, for balanced battery usage?", "Set a reminder for 'Mom Birthday Party' on '2024-01-10-18-00' to '2024-01-10-21-00'.", "Organize a 'Community Clean-up' starting at '2023-11-15-09-00' and ending at '2023-11-15-12-00'.", "Plan a romantic dinner event titled 'Anniversary Dinner' from 8 PM to 10 PM on December 12, 2023", "I want to add a 'Book Club Meeting' event from '2023-09-05-19-00' to '2023-09-05-20-30'.", "What's the proper way to set up a 'Coffee Break' event starting at 2023-09-05-15-00 and finishing at 2023-09-05-15-15?", "Schedule a team meeting for project kickoff with'start_time at 2023-10-01-09-00 and end_time at 2023-10-01-10-00'", "Add a personal event titled 'Mortgage Meeting at the Bank' beginning at 2023-04-18-12-00 and ending at 2023-04-18-13-30.", "Create a 'Yoga Session' calendar event that starts at '2023-06-08-06-00' and finishes at '2023-06-08-07-00'.", "Can you schedule a calendar event with the title 'Team Meeting' starting on '2023-05-15-14-00' and ending at '2023-05-15-15-00'?", "I need to add 'Lunch with Sarah' to my calendar, starting on 2023-07-22-13-00 and ending on 2023-07-22-14-00. How do I do it?", 'Set the media volume to 2 advised for background music while working', 'Can the alarm volume be set to a comfortable 4? I prefer waking up gently.', 'Can you set the media volume to 5?', "I'm at a party and want my media volume at full blast. Set it to 10, please.", 'Adjust the media volume to 8 for an immersive movie watching experience tonight.', 'Could you set the alarm volume to 5 for a moderate wake-up experience?', 'For a better morning wake-up, set the alarm volume to 8.', "Can you set my alarm volume to 5? I think that's a good level to wake me up without being too loud.", 'Could you set the media volume to 5 for a balanced audio experience during my movie tonight?', "I'd like to mute the media volume while at work.", "I'm having trouble pairing with 'Galaxy Buds 2023'. Could you attempt to connect within timeout 10?", "Could we connect to 'Bose QuietComfort 35 II'? Please try with a longer timeout of 45 seconds.", "Try connecting to 'SmartHomeHub' with a 12-second timeout to see if a slightly longer duration improves the success rate.", "Could you guide on connecting to 'Google Nest Audio' with a timeout of 10 seconds, to test if it connects within the default time limit?", "Let's see if 'HeadphonesXYZ' can be swiftly connected within just 4 seconds.", "Is it possible to establish a connection with 'Beats Studio 3' within just 5 seconds?", "Looking to connect to my car's Bluetooth system 'Tesla Model S' with timeout of 10s. Is it feasible?", "Can you connect to 'JBL Flip 4' speaker within a 20-second timeout period?", "Can you try connecting to my Bose SoundLink speaker? I believe it's named 'Bose-SL'. Please give it up to 20 seconds.", "Try connecting to 'UE Boom 3' with the shortest possible timeout. How about 3 seconds?", 'Could you switch on Do Not Disturb mode during my bedtime hours from 10 PM to 7 AM?', 'How do I disable the Do Not Disturb feature on my device?', 'Is it possible to enable Do Not Disturb mode automatically at 10 PM every day?', 'I need to concentrate on my studies. Turn on Do Not Disturb mode for the next three hours.', "I'm heading into a meeting. Activate Do Not Disturb mode, please.", 'How do I switch back from Do Not Disturb mode to receive all notifications?', 'Please show me how to switch off Do Not Disturb mode using a function call.', "I'm starting my study session, can you enable Do Not Disturb to avoid distractions?", "I'm entering a meeting; could you enable Do Not Disturb?", 'Before my nap, can you ensure Do Not Disturb is on?', 'For my plants, keep the temperature at a steady 69 degrees.', 'Can you set the Nest Thermostat to a cozy 72 degrees for the evening?', "Before we arrive home, adjust the Nest Thermostat to 71 degrees so it's nice and warm.", 'Help me save on my energy bill by setting the Nest temperature to 78 degrees.', 'For a perfect winter morning, could the Nest be set to 70 degrees Fahrenheit?', "I'm feeling a bit chilly, could you adjust my thermostat to 75 degrees?", "To avoid freezing pipes while I'm away, set the thermostat to 50 degrees Fahrenheit, please.", 'For a cozy evening, could you increase the Nest thermostat temperature to 75 degrees Fahrenheit?', 'My plants are sensitive to cold; can the living area be maintained at 77 degrees?', 'Adjust the Nest to 70 degrees, please. I want to keep the house comfortable for the guests arriving.', "How do I get '80s Smash Hits' to play on my Google Nest Hub from Amazon Music?", "Could you stream '80s Smash Hits' from Apple Music on my Nest Hub, please?", 'Is it possible to stream live jazz music from Tidal on my Google Nest Hub?', "Play 'Rock Classics' on Deezer through my Nest Audio. Set music service to 'Deezer' and music name to 'Rock Classics'.", "Can you start streaming 'Country Gold' playlist from Google Play Music on my Nest Hub?", "I'd love to hear Taylor Swift's 'Folklore' album on Apple Music through my Nest Audio device.", "I want to listen to 'Jazz Classics' from Tidal on my Google Nest. How can I do that?", "I'd like 'Classical Music for Reading' to play on Napster via my Google Nest Audio. Is that possible?", "Can you play the 'Top Hits 2023' playlist from Spotify on my Nest Hub?", "I would love to hear '90s Hip Hop' music from Tidal on my Nest Hub.", "Is it possible to play the 'Top 50 Global' playlist from Spotify video on my Google Nest Hub?", "I'd like to watch 'MasterChef' episodes from Hulu on my Google Home device, please.", "What's the procedure to watch 'Planet Earth II' series from Netflix on my Nest Hub?", "I'd like to watch 'MasterChef' series from Amazon Prime Video on my Nest Hub.", "How to stream 'Latest Tech Gadgets' videos from YouTube on my Nest Hub?", "I'd like to watch 'MasterChef' episodes from Hulu on my Google Home device, please.", "How do I watch 'The Crown' series from Netflix on my Nest Hub?", "Can I watch 'Breaking Bad' from Netflix on my Nest Hub? What are the steps?", "Please, play workout videos from 'FitnessBlender' on YouTube on my Nest Hub.", "Play the 'Workout Mix' video playlist from Apple Music on my Google Nest Hub.", "Can you open the front door? I've got my hands full.", "Could you lock all doors? I'm leaving for vacation and want to ensure everything is secured.", "Can you check if the backdoor is locked and lock it if it's not?", 'Please lock all doors at 10 pm every night for security.', "After leaving for vacation, I can't remember if I locked the door. Could you secure it now?", "I think I forgot to lock the back door, can you check and lock it if it's open?", 'Please lock the back door; I think I forgot to do it myself.', 'For the party tonight, please set the back door to unlock automatically when guests arrive.', 'Please lock all doors at 10 pm every night for security.', 'We are going on vacation and want to make sure the house is secure. Could you double-check if all doors are locked?', 'Receive trending news about artificial intelligence in Korean', 'Fetches latest trending news in the technology sector, specifically in Spanish', "Need the latest trending articles on 'education' in Arabic.", 'Find trending news related to artificial intelligence in Russian', 'Discover trending news about the global economy in Mandarin', 'Retrieves recent trending articles related to COVID-19 vaccines available in French', "I'd like to read about 'global economy' in the Russian language", 'Look up the latest sports news in the world of football in German', 'Look for trending news on cryptocurrency market movements, with information provided in Japanese', 'Request trending news articles about climate change in English', 'Improve battery life with power saving tips.', 'View battery health and optimize for longer usage.', 'Find and install new keyboard themes.', 'How to use split-screen mode?', 'Find the best meditation apps available on Google Play.', 'Find and install new keyboards from the Google Play Store.', 'Check available storage space.', 'Translate text using Google Translate.', 'How to block notifications from a specific app?', 'Optimize storage space on device.', "I'm looking for the upcoming weather in Cairo, Egypt, can you help?", 'What is the expected weather in Tokyo, Japan, for the next three days?', "What's the weather forecast for the next 6 days in Mumbai?", 'Can you provide the weather situation in Dubai for the upcoming week, especially any heatwave alerts?', "What's the weather like in New York City for the next three days?", 'Show me the weather forecast in Tokyo for the next three days, please.', "I'm looking for the 7-day weather outlook for 90210 ZIP code.", 'I need to know the weather forecast for San Francisco for the next two days, particularly any fog warnings.', 'Can I get a 7-day weather forecast for Toronto? Need it for a planned vacation.', 'Is it going to be sunny in Rio de Janeiro during the Carnival next year?', "I need to send an urgent email to hr@ourcompany.com with the title 'Immediate Resignation Notice' expressing my regret and providing two weeks' notice, including today's date as well.", "Can you send an email to jane.doe@example.com with the title 'Meeting Update' and content discussing the new meeting schedule for next week?", "Can you send an email to john.doe@example.com with the title 'Project Update' and content regarding the latest progress and next steps for our project?", "Can you send an email to john.doe@example.com with the title 'Meeting Update' and content regarding the rescheduling of the Wednesday meeting to Friday at 10 am?", "Dispatch a thank you email to speaker@example.com with the title 'Appreciation Note' for their inspiring talk at our recent event, mentioning key takeaways.", "Can you send an email to john.doe@example.com with the title 'Meeting Update' and content regarding the rescheduling of the Wednesday meeting to Friday at 10 am?", 'Send a weekly newsletter to a subscriber with email \'subscriber@example.com\' and title \'Your Weekly Newsletter\' and content "Here\'s what\'s new this week!"', "Can I send a reminder email to team@example.com with the title 'Project Deadline Reminder' and include the final deadline details in the content?", "Notify a customer about their order shipment with recipient 'customer123@example.com' and title 'Your Order Has Been Shipped!' and content 'Hello! Your order #456789 has been dispatched and will reach you within 3-5 working days.'", "Send seasonal greetings to clients and partners with recipient 'valuedclient@example.com' and title 'Happy Holidays from All of Us!' and content 'Wishing you a joyous holiday season and a prosperous New Year. Thanks for your continued support.'", "I'm looking for YouTube tutorials on Python programming for beginners.", 'Can you find any YouTube documentaries on ancient civilizations?', 'Search for comedy sketches for a good laugh.', 'Can you find me the top 5 educational channels for learning Python?', 'Search for comedy sketches for a good laugh.', 'Highlights from the 2023 World Cup final', 'Can you find the most watched stand-up comedy videos this year?', 'Find me documentaries on climate change released this year.', 'Looking for live performances by BTS from their latest tour.', "Find 'live coding streams' happening now on YouTube.", 'How to get from La Sagrada Familia to Park Guell by bike?', 'Transit directions from Tokyo Station to Shibuya Crossing.', 'Find a driving route from Golden Gate Bridge to Stanford University.', 'How do I walk from the Louvre Museum to the Eiffel Tower?', "What's the best bicycling path from Stanford University to Googleplex?", 'What are the walking routes available from Buckingham Palace to the British Museum?', 'Show the walking route from the Louvre to the Eiffel Tower.', 'What is the quickest public transit route from Shibuya to Tokyo Tower?', 'Find a walking route from Sydney Opera House to the Sydney Harbour Bridge with the least amount of stairs.', 'Could you find a walking route that takes me from Buckingham Palace to the British Museum?', 'How do I check if the battery of my smoke detector needs replacing?', 'After replacing my WiFi router, I need to confirm if my Nest Smoke Detector is still connected. Can you verify?', 'What is the current battery status of my Nest Smoke Detector in the guest room?', 'check my Nest Smoke Detector is functioning optimally', 'Is my Nest Smoke Detector on the main floor operating correctly, including its connectivity and sensor status?', "I'd like to know if all the sensors in my smoke detector are functioning correctly.", 'Can I obtain the overall health status of my Nest Smoke Detector without technical expertise?', 'Check function regarding smoke detector maintenance?', 'I need guidance on assessing the connectivity status of my Nest Smoke Detector. How do I go about it?', "Is my Nest Smoke Detector connected properly to my network? I'm looking for a connectivity status report.", "How to send a 'Thinking of you!' text to my contact named Alex Reed?", "Please send 'Do we need anything from the grocery store?' message to Noah Brown.", "Please dispatch a text to 'Julia' with 'Miss you! Let's catch up over coffee this weekend?'", "Please send a text to 'John Smith' saying 'Meeting is rescheduled to 3 PM, let everyone know.'", "How do I send a 'Congratulations on your newborn!' message to Lisa?", "Send a text to 'Emergency Contact' with 'Had a minor car accident. I'm okay, but will be late.'", "Inform Alice about the project delay by sending her 'Project deadline pushed to next Monday. Let's catch up tomorrow to discuss details.'", "Send a 'Good luck on your presentation today! You'll do great!' message to Kevin.", "Can you send a text to the contact named Sam? It should read 'The game starts at 8 PM. Are we still meeting at yours?'", "Send a message to Dr. Smith, 'I wanted to confirm my appointment for this Friday at 10 AM. Is that still okay?'", "I'm trying to reach a client overseas. The number is +913322667890. Can we initiate a call to this number?", "Is it possible to start a phone call to my colleague's international number, which is +919876543210?", "Make a phone call to '+12345678900,555' where '555' is the extension", 'I lost my phone; can you call my number (+11234567890) so I can find it by the ring?', 'Make a phone call to a UK phone number +447700900000', 'Is it possible to make a phone call to my aunt who lives abroad? Her phone number is +33550123456.', 'Could you help me connect to a local restaurant? Their number is 4041234567.', "There's a delivery at the door. Call the number on the invoice, please. It's +44800526123.", "{'description': 'Make a phone call to a friend in the US.', 'arguments': {'phone_number': '+11234567890'}}", 'Could you demonstrate making a call to a standard US number like +13015550101 for customer service support?', 'Could you create a contact for my new colleague? Her name is Sarah Brown and her number is +12309876543.', "Let's save the number of the new Italian restaurant I found. Name: Bella Italia, Phone: +15556667777.", 'Add my new colleague, Lisa Ray, to my phone book. Her contact number is +17778078965.', "Let's save the number of the new Italian restaurant I found. Name: Bella Italia, Phone: +15556667777.", 'Could you create a contact for my dentist, Dr. Aaron Shaw? His phone number is +16665432109.', "What's the correct way to create a contact entry for my dentist, Dr. Emily Stone, with her number as +15005550006?", 'Could you create a contact for my boss, Mr. David Anderson? His contact number is +12347654321.', 'Please create a contact entry for Emily Clark, phone number being +12345678901.', 'Can you add John Doe with the phone number +11234567890 to my address book?', 'Store the contact info of my personal trainer, Alex Strong, with their number being +10987654321.'],
        help="List of prompts to use for benchmarking. Separate multiple prompts with spaces.",
    )

    args = parser.parse_args()

    verbose = args.verbose
    skip_models = args.skip_models
    prompts = args.prompts
    print(
        f"\nVerbose: {verbose}\nSkip models: {skip_models}\nPrompts: {prompts}"
    )

    model_names = get_benchmark_models(skip_models)
    benchmarks = {}

    for model_name in model_names:
        responses: List[OllamaResponse] = []
        for prompt in prompts:
            if verbose:
                print(f"\n\nBenchmarking: {model_name}\nPrompt: {prompt}")
            response = run_benchmark(model_name, prompt, verbose=verbose)
            responses.append(response)

            if verbose:
                print(f"Response: {response.message.content}")
                inference_stats(response)
        benchmarks[model_name] = responses

    for model_name, responses in benchmarks.items():
        average_stats(responses)


if __name__ == "__main__":
    main()
    # Example usage:
    # python benchmark.py --verbose --skip-models aisherpa/mistral-7b-instruct-v02:Q5_K_M llama2:latest --prompts "What color is the sky" "Write a report on the financials of Microsoft"
