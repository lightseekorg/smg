use anyhow::{anyhow, bail, Result};
use smg_skills::{parse_skill_bundle, ParsedSkillBundle, SkillParseError, SkillParseWarningKind};

fn parse_valid_bundle(openai_yaml: Option<&str>) -> Result<ParsedSkillBundle, SkillParseError> {
    parse_skill_bundle(
        r"---
name: github:gh-fix-ci
description: Diagnose and fix CI failures for an attached PR.
metadata:
  short-description: Base short description from SKILL.md
---

## Instructions

Read the failing logs first.
",
        openai_yaml,
    )
}

#[test]
fn parses_skill_md_and_projects_sidecar_short_description() -> Result<()> {
    let parsed = parse_valid_bundle(Some(
        r##"
interface:
  display_name: GitHub CI Fixer
  short_description: Diagnose failing GitHub Actions runs
  icon_small: assets/icon-small.png
  icon_large: assets/icon-large.png
  brand_color: "#1D4ED8"
  default_prompt: Start with the failing workflow run.
dependencies:
  tools:
    - type: mcp
      value: github
      description: GitHub access
      transport: streamable_http
      url: https://example.invalid/mcp
policy:
  allow_implicit_invocation: true
  products:
    - codex
    - smg
"##,
    ))?;

    assert_eq!(parsed.name, "github:gh-fix-ci");
    assert_eq!(
        parsed.description,
        "Diagnose and fix CI failures for an attached PR."
    );
    assert_eq!(
        parsed.short_description.as_deref(),
        Some("Diagnose failing GitHub Actions runs")
    );
    assert_eq!(
        parsed.instructions_body,
        "\n## Instructions\n\nRead the failing logs first.\n"
    );
    assert!(parsed.warnings.is_empty());

    let interface = parsed
        .interface
        .ok_or_else(|| anyhow!("expected interface metadata"))?;
    assert_eq!(interface.display_name.as_deref(), Some("GitHub CI Fixer"));
    assert_eq!(
        interface.icon_small.as_deref(),
        Some("assets/icon-small.png")
    );
    assert_eq!(interface.brand_color.as_deref(), Some("#1D4ED8"));

    let dependencies = parsed
        .dependencies
        .ok_or_else(|| anyhow!("expected dependency metadata"))?
        .tools;
    assert_eq!(dependencies.len(), 1);
    assert_eq!(dependencies[0].tool_type, "mcp");
    assert_eq!(dependencies[0].value, "github");

    let policy = parsed
        .policy
        .ok_or_else(|| anyhow!("expected policy metadata"))?;
    assert_eq!(policy.allow_implicit_invocation, Some(true));
    assert_eq!(policy.products, vec!["codex", "smg"]);
    Ok(())
}

#[test]
fn rejects_skill_md_missing_name() -> Result<()> {
    let error = match parse_skill_bundle(
        r"---
description: Diagnose and fix CI failures for an attached PR.
---
instructions
",
        None,
    ) {
        Ok(bundle) => bail!("missing name must fail, got {bundle:?}"),
        Err(error) => error,
    };

    assert_eq!(
        error,
        SkillParseError::MissingRequiredField { field: "name" }
    );
    Ok(())
}

#[test]
fn rejects_skill_md_missing_description() -> Result<()> {
    let error = match parse_skill_bundle(
        r"---
name: github:gh-fix-ci
---
instructions
",
        None,
    ) {
        Ok(bundle) => bail!("missing description must fail, got {bundle:?}"),
        Err(error) => error,
    };

    assert_eq!(
        error,
        SkillParseError::MissingRequiredField {
            field: "description"
        }
    );
    Ok(())
}

#[test]
fn rejects_invalid_frontmatter_yaml() -> Result<()> {
    let error = match parse_skill_bundle(
        r"---
name: github:gh-fix-ci
description: [unterminated
---
instructions
",
        None,
    ) {
        Ok(bundle) => bail!("invalid frontmatter YAML must fail, got {bundle:?}"),
        Err(error) => error,
    };

    match error {
        SkillParseError::InvalidFrontmatterYaml { message } => {
            assert!(!message.is_empty());
        }
        other => bail!("expected invalid frontmatter YAML error, got {other:?}"),
    }
    Ok(())
}

#[test]
fn ignores_invalid_sidecar_yaml_at_file_level() -> Result<()> {
    let parsed = parse_valid_bundle(Some("interface: [broken"))?;

    assert_eq!(
        parsed.short_description.as_deref(),
        Some("Base short description from SKILL.md")
    );
    assert!(parsed.interface.is_none());
    assert!(parsed.dependencies.is_none());
    assert!(parsed.policy.is_none());
    assert_eq!(parsed.warnings.len(), 1);
    assert_eq!(
        parsed.warnings[0].kind,
        SkillParseWarningKind::SidecarFileIgnored
    );
    assert_eq!(parsed.warnings[0].path, "agents/openai.yaml");
    Ok(())
}

#[test]
fn rejects_reserved_namespace_segment_in_name() -> Result<()> {
    let error = match parse_skill_bundle(
        r"---
name: anthropic:gh-fix-ci
description: Diagnose and fix CI failures for an attached PR.
---
instructions
",
        None,
    ) {
        Ok(bundle) => bail!("reserved namespace segment must fail, got {bundle:?}"),
        Err(error) => error,
    };

    assert_eq!(
        error,
        SkillParseError::InvalidField {
            field: "name",
            message: "is reserved".to_owned(),
        }
    );
    Ok(())
}

#[test]
fn accepts_character_based_description_length_limit() -> Result<()> {
    let description = "😀".repeat(1024);
    let skill_md =
        format!("---\nname: github:gh-fix-ci\ndescription: {description}\n---\ninstructions\n");

    let parsed = parse_skill_bundle(&skill_md, None)?;
    assert_eq!(parsed.description.chars().count(), 1024);
    Ok(())
}

#[test]
fn accepts_dot_slash_prefixed_icon_paths() -> Result<()> {
    let parsed = parse_valid_bundle(Some(
        r"
interface:
  icon_small: ./assets/icon-small.png
  icon_large: ./assets/icon-large.png
",
    ))?;

    let interface = parsed
        .interface
        .ok_or_else(|| anyhow!("expected interface metadata"))?;
    assert_eq!(
        interface.icon_small.as_deref(),
        Some("./assets/icon-small.png")
    );
    assert_eq!(
        interface.icon_large.as_deref(),
        Some("./assets/icon-large.png")
    );
    assert!(parsed.warnings.is_empty());
    Ok(())
}

#[test]
fn does_not_treat_nested_angle_brackets_as_xml_tag() -> Result<()> {
    let parsed = parse_skill_bundle(
        r"---
name: github:gh-fix-ci
description: Compare <emphasis and < later > symbols literally.
---
instructions
",
        None,
    )?;

    assert_eq!(
        parsed.description,
        "Compare <emphasis and < later > symbols literally."
    );
    Ok(())
}

#[test]
fn rejects_actual_xml_like_tag_in_description() -> Result<()> {
    let error = match parse_skill_bundle(
        r"---
name: github:gh-fix-ci
description: Render <em>highlighted</em> text.
---
instructions
",
        None,
    ) {
        Ok(bundle) => bail!("xml-like tag must fail, got {bundle:?}"),
        Err(error) => error,
    };

    assert_eq!(
        error,
        SkillParseError::InvalidField {
            field: "description",
            message: "must not contain XML-like tags".to_owned(),
        }
    );
    Ok(())
}

#[test]
fn salvages_valid_sidecar_fields_after_successful_parse() -> Result<()> {
    let parsed = parse_valid_bundle(Some(
        r"
interface:
  display_name: GitHub CI Fixer
  short_description: 7
  icon_small: ../assets/icon-small.png
  icon_large: assets/icon-large.png
  brand_color: blue
  default_prompt: Start with the failing workflow run.
dependencies:
  tools:
    - type: mcp
      value: github
      description: GitHub access
    - type: 42
      value: broken
    - value: missing-type
policy:
  allow_implicit_invocation: true
  products:
    - codex
    - 123
",
    ))?;

    assert_eq!(
        parsed.short_description.as_deref(),
        Some("Base short description from SKILL.md")
    );

    let interface = parsed
        .interface
        .ok_or_else(|| anyhow!("expected salvaged interface"))?;
    assert_eq!(interface.display_name.as_deref(), Some("GitHub CI Fixer"));
    assert_eq!(
        interface.icon_large.as_deref(),
        Some("assets/icon-large.png")
    );
    assert_eq!(
        interface.default_prompt.as_deref(),
        Some("Start with the failing workflow run.")
    );
    assert!(interface.short_description.is_none());
    assert!(interface.icon_small.is_none());
    assert!(interface.brand_color.is_none());

    let dependencies = parsed
        .dependencies
        .ok_or_else(|| anyhow!("expected salvaged dependency metadata"))?
        .tools;
    assert_eq!(dependencies.len(), 1);
    assert_eq!(dependencies[0].tool_type, "mcp");
    assert_eq!(dependencies[0].value, "github");

    let policy = parsed
        .policy
        .ok_or_else(|| anyhow!("expected salvaged policy metadata"))?;
    assert_eq!(policy.allow_implicit_invocation, Some(true));
    assert_eq!(policy.products, vec!["codex"]);

    assert_eq!(parsed.warnings.len(), 6);
    assert!(parsed
        .warnings
        .iter()
        .all(|warning| warning.kind == SkillParseWarningKind::SidecarFieldIgnored));
    Ok(())
}
