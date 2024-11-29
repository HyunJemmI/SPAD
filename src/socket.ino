#include <SPI.h>
#include <Ethernet.h>
#include <PubSubClient.h>

// MQTT Broker 정보 설정
const char* mqtt_server = "192.168.0.9";
const int mqtt_port = 1883;
const char* mqtt_username = "spad";
const char* mqtt_password = "spad";

// MQTT 클라이언트 설정
EthernetClient ethClient;
PubSubClient client(ethClient);

byte mac[] = { 0x00, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE }; // 예시 MAC 주소


void setup() {
  // 네트워크 초기화 및 연결
  Ethernet.begin(mac); // 이더넷 쉴드를 사용하는 경우 mac 주소 설정
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);

  // 연결 시도
  reconnect();
}

void loop() {
  // MQTT 클라이언트 유지
  if (!client.connected()) {
    reconnect();
  }

  // 다른 작업 수행

  // MQTT 메시지 발행 예시
  client.publish("topic명", "메시지 내용");

  // MQTT 메시지 수신 및 처리 예시
  client.loop();
}

void callback(char* topic, byte* payload, unsigned int length) {
  // 메시지 수신 시 실행되는 콜백 함수
  // payload를 해석하고 필요한 작업을 수행
}

void reconnect() {
  // MQTT 브로커에 재연결 시도
  while (!client.connected()) {
    Serial.print("연결 중...");
    if (client.connect("아두이노_클라이언트_ID", mqtt_username, mqtt_password)) {
      Serial.println("연결 성공");
      client.subscribe("topic명");
    } else {
      Serial.print("연결 실패, rc=");
      Serial.println(client.state());
      delay(5000);
    }
  }
}