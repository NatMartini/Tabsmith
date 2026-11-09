//! `model!(…, rule(…))`: rules proved when the macro expands, their proofs checked again by the
//! compiler. That these modules compile is most of the test.

tabsmith_macros::model!(
    defects,
    "../../fixtures/bundles/binary-gbdt",
    rule(hours_in_service in 0..=1000, _ within 2 sd => probability <= 0.82),
    rule(_ within 2 sd, line == "A" => probability >= 0.01),
    rule(_ within 1 sd => probability <= 0.85),
);

tabsmith_macros::model!(
    defects_arrays,
    "../../fixtures/bundles/binary-gbdt",
    tree_arrays,
    rule(hours_in_service in 0..=1000, _ within 2 sd => probability <= 0.82),
);

tabsmith_macros::model!(
    faults,
    "../../fixtures/bundles/multiclass-gbdt",
    rule(_ within 0.3 sd, line == "C", shift == "night", supplier == "S4", device_type == "valve",
         firmware == "v2.0", maintenance_flag == true => label != "seal"),
    rule(_ within 0.05 sd, line == "A", shift == "day", supplier == "S1", device_type == "pump",
         firmware == "v1.3", maintenance_flag == false => label == "none"),
);

tabsmith_macros::model!(hours, "../../fixtures/bundles/regression-mlp", rule(_ within 2 sd => value <= 100000));

#[test]
fn the_module_lists_its_rules_in_their_canonical_text() {
    assert_eq!(
        defects::RULES,
        [
            "hours_in_service in 0..=1000, _ within 2 sd => probability <= 0.82",
            "_ within 2 sd, line == \"A\" => probability >= 0.01",
            "_ within 1 sd => probability <= 0.85",
        ]
    );
    assert_eq!(defects_arrays::RULES, [defects::RULES[0]]);
    assert_eq!(faults::RULES.len(), 2);
    assert_eq!(hours::RULES, ["_ within 2 sd => value <= 100000"]);
}

#[test]
fn inputs_of_a_rules_region_keep_its_claim() {
    // The examples moved into the first rule's region: few hours in service, the other numbers
    // at their training means.
    for (i, e) in defects::EXAMPLES.iter().enumerate() {
        let at_means = defects::Input {
            hours_in_service: Some(500.0),
            cycles: Some(1009.53),
            temp_mean: Some(61.88),
            temp_std: Some(2.1),
            temp_max: Some(68.34),
            vibration_rms: Some(0.98),
            vibration_peak: Some(3.06),
            pressure_mean: Some(4.19),
            pressure_std: Some(0.3),
            current_mean: Some(11.63),
            current_std: Some(0.81),
            humidity: Some(45.02),
            voltage_drop: Some(0.8),
            ambient_noise_db: Some(49.92),
            ..*e
        };
        let p = defects::predict(&at_means).unwrap();
        assert!(p.probability <= 0.82, "example {i}: {}", p.probability);
        assert_eq!(
            defects_arrays::predict(&defects_arrays::Input::from_values(&at_means.values()).unwrap())
                .unwrap()
                .probability,
            p.probability
        );
    }
}
