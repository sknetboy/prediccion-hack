package com.churninsight.controllers;

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
class StatsControllerTest {
    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private RestTemplate restTemplate;

    @Test
    void stats_ok() throws Exception {
        MockRestServiceServer server = MockRestServiceServer.createServer(restTemplate);
        server.expect(MockRestRequestMatchers.requestTo("http://127.0.0.1:8001/stats"))
                .andRespond(MockRestResponseCreators.withSuccess("{\"total_evaluados\":1,\"tasa_churn\":0.0}", MediaType.APPLICATION_JSON));

        mockMvc.perform(org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get("/stats"))
                .andExpect(org.springframework.test.web.servlet.result.MockMvcResultMatchers.status().isOk())
                .andExpect(org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath("$.total_evaluados").value(1))
                .andExpect(org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath("$.tasa_churn").value(0.0));

        server.verify();
    }
}