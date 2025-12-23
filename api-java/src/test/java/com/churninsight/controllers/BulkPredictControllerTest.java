package com.churninsight.controllers;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.test.web.client.match.MockRestRequestMatchers;
import org.springframework.test.web.client.response.MockRestResponseCreators;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.web.client.RestTemplate;

@SpringBootTest
@AutoConfigureMockMvc
public class BulkPredictControllerTest {
    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private RestTemplate restTemplate;

    @Test
    void testBulkForward() throws Exception {
        MockRestServiceServer server = MockRestServiceServer.createServer(restTemplate);
        server.expect(MockRestRequestMatchers.requestTo("http://127.0.0.1:8001/predict_bulk"))
                .andRespond(MockRestResponseCreators.withSuccess("[{\\\"prevision\\\":\\\"Va a continuar\\\",\\\"probabilidad\\\":0.32}]", MediaType.APPLICATION_JSON));

        MockMultipartFile file = new MockMultipartFile("file", "bulk.csv", "text/csv", "a,b\n1,2".getBytes());
        mockMvc.perform(org.springframework.test.web.servlet.request.MockMvcRequestBuilders
                        .multipart("/predict-bulk").file(file))
                .andExpect(org.springframework.test.web.servlet.result.MockMvcResultMatchers.status().isOk());
    }
}