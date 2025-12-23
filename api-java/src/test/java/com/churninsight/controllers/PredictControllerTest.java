package com.churninsight.controllers;

import com.churninsight.dto.ClientData;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.test.web.client.match.MockRestRequestMatchers;
import org.springframework.test.web.client.response.MockRestResponseCreators;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.web.client.RestTemplate;

@SpringBootTest
@AutoConfigureMockMvc
class PredictControllerTest {
    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private RestTemplate restTemplate;

    @Test
    void predict_ok() throws Exception {
        MockRestServiceServer server = MockRestServiceServer.createServer(restTemplate);
        server.expect(MockRestRequestMatchers.requestTo("http://127.0.0.1:8001/predict"))
                .andRespond(MockRestResponseCreators.withSuccess("{\"prevision\":\"Va a continuar\",\"probabilidad\":0.32,\"explicacion\":[\"plan\",\"retrasos_pago\",\"tiempo_contrato_meses\"]}", MediaType.APPLICATION_JSON));

        String body = "{" +
                "\"tiempo_contrato_meses\":12," +
                "\"retrasos_pago\":2," +
                "\"uso_mensual\":14.5," +
                "\"plan\":\"Premium\"," +
                "\"nps\":7," +
                "\"quejas\":1," +
                "\"canal_contacto\":\"app\"," +
                "\"interacciones_soporte\":2," +
                "\"tipo_pago\":\"debito_automatico\"," +
                "\"region\":\"norte\"," +
                "\"tipo_cliente\":\"nuevo\"" +
                "}";

        mockMvc.perform(org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post("/predict")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body))
                .andExpect(org.springframework.test.web.servlet.result.MockMvcResultMatchers.status().isOk())
                .andExpect(org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath("$.prevision").value("Va a continuar"))
                .andExpect(org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath("$.probabilidad").value(0.32))
                .andExpect(org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath("$.explicacion[0]").value("plan"));

        server.verify();
    }
}