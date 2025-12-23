package com.churninsight.controllers;

import java.util.Map;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

@RestController
@RequestMapping("/calibrate")
public class CalibrateController {
    private final RestTemplate restTemplate;

    public CalibrateController(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    @Value("${python.service.calibrate.url:http://127.0.0.1:8001/calibrate_threshold}")
    private String pythonCalibrateUrl;

    @PostMapping
    public ResponseEntity<Map> calibrate(@RequestParam(required = false, defaultValue = "f1") String modo,
                                         @RequestParam(required = false, defaultValue = "40") Double beneficio,
                                         @RequestParam(required = false, defaultValue = "10") Double costo) {
        String url = UriComponentsBuilder.fromHttpUrl(pythonCalibrateUrl)
                .queryParam("modo", modo)
                .queryParam("beneficio", beneficio)
                .queryParam("costo", costo)
                .build().toUriString();
        Map resp = restTemplate.postForObject(url, null, Map.class);
        return ResponseEntity.ok(resp);
    }
}