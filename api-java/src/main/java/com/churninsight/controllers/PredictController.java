package com.churninsight.controllers;

import com.churninsight.dto.ClientData;
import com.churninsight.dto.PredictResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

@RestController
@RequestMapping("/predict")
public class PredictController {
    private final RestTemplate restTemplate;

    public PredictController(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    @Value("${python.service.url:http://127.0.0.1:8001/predict}")
    private String pythonServiceUrl;

    @PostMapping
    public ResponseEntity<PredictResponse> predict(@Validated @RequestBody ClientData payload,
                                                   @RequestParam(required = false) Double umbral) {
        UriComponentsBuilder builder = UriComponentsBuilder.fromHttpUrl(pythonServiceUrl);
        if (umbral != null) {
            builder.queryParam("umbral", umbral);
        }
        String url = builder.build().toUriString();
        PredictResponse resp = restTemplate.postForObject(url, payload, PredictResponse.class);
        return ResponseEntity.ok(resp);
    }
}