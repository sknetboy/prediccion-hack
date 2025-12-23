package com.churninsight.controllers;

import java.util.Map;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

@RestController
@RequestMapping("/retrain")
public class RetrainController {
    private final RestTemplate restTemplate;

    public RetrainController(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    @Value("${python.service.retrain.url:http://127.0.0.1:8001/retrain}")
    private String pythonRetrainUrl;

    @PostMapping
    public ResponseEntity<Map> retrain(@RequestParam(required = false) String data_path) {
        String url = pythonRetrainUrl;
        if (data_path != null && !data_path.isBlank()) {
            url = UriComponentsBuilder.fromHttpUrl(pythonRetrainUrl)
                    .queryParam("data_path", data_path)
                    .build().toUriString();
        }
        Map resp = restTemplate.postForObject(url, null, Map.class);
        return ResponseEntity.ok(resp);
    }
}