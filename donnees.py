import paho.mqtt.client as mqtt
import json


def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    client.subscribe("AM107/#")


def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))
    payload = json.loads(msg.payload)

    line = ""

    for data in payload[0]:
        line +=  + ";"
    print(line)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("chirpstack.iut-blagnac.fr", 1883, 60)

client.loop_forever()