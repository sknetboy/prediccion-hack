package com.churninsight.controllers;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.util.UriComponentsBuilder;

@RestController
@RequestMapping("/predict-bulk")
public class BulkPredictController {
    private final RestTemplate restTemplate;

    public BulkPredictController(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    @Value("${python.service.bulk.url:http://127.0.0.1:8001/predict_bulk}")
    private String pythonBulkUrl;

    @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<String> predictBulk(@RequestParam("file") MultipartFile file,
                                              @RequestParam(required = false) Double umbral) throws Exception {
        String url = pythonBulkUrl;
        if (umbral != null) {
            url = UriComponentsBuilder.fromHttpUrl(pythonBulkUrl).queryParam("umbral", umbral).build().toUriString();
        }
        ByteArrayResource resource = new ByteArrayResource(file.getBytes()) {
            @Override
            public String getFilename() { return file.getOriginalFilename(); }
        };
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", resource);
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        HttpEntity<MultiValueMap<String, Object>> entity = new HttpEntity<>(body, headers);
        ResponseEntity<String> resp = restTemplate.postForEntity(url, entity, String.class);
        return ResponseEntity.status(resp.getStatusCode()).body(resp.getBody());
    }
}