//! Tool schema shapes per wire protocol.
//!
//! [`codex_tools::ToolSpec`] is the canonical description of a tool, and its
//! own serialization is the shape the Responses API expects. The other
//! protocols carry one flat function name per tool and have no namespace field,
//! so a namespaced tool has to be flattened before it can cross over, and the
//! flat name has to be mapped back when the model calls it. Each protocol then
//! gets its own serializer for the flat form rather than a second copy of the
//! tool definitions.

use std::collections::BTreeMap;
use std::collections::BTreeSet;

use async_openai::types::chat::ChatCompletionTool;
use async_openai::types::chat::ChatCompletionTools;
use async_openai::types::chat::FunctionObject;
use claudius::ToolParam;
use codex_protocol::DEFAULT_FUNCTION_NAMESPACE;
use codex_protocol::ToolName;
use codex_tools::JsonSchema;
use codex_tools::ResponsesApiNamespaceTool;
use codex_tools::TOOL_SEARCH_TOOL_NAME;
use codex_tools::ToolSpec;
use serde_json::Value;

use crate::ProviderError;

/// The longest tool name these protocols accept. Function names are capped at
/// 64 characters, and a request that carries a longer one is rejected whole.
const MAX_WIRE_TOOL_NAME_BYTES: usize = 64;

/// The name one wire protocol carries for a tool.
///
/// A namespace that already ends in a delimiter, which is the shape MCP
/// namespaces have, is concatenated; any other namespace is joined with `__` so
/// `web` + `run` reads as `web__run` rather than `webrun`. A name longer than
/// these protocols accept keeps its head and ends in a digest of the whole
/// name, which keeps distinct tools distinct and the shortening stable.
pub fn wire_tool_name(namespace: Option<&str>, name: &str) -> String {
    let flattened = match namespace {
        None | Some("") | Some(DEFAULT_FUNCTION_NAMESPACE) => name.to_string(),
        Some(namespace) if namespace.ends_with('_') => format!("{namespace}{name}"),
        Some(namespace) => format!("{namespace}__{name}"),
    };
    fit_wire_name(flattened)
}

/// Shortens a name to the length these protocols accept, digesting what is cut
/// so two long names sharing a prefix stay apart. A name that already fits is
/// returned as it is.
fn fit_wire_name(name: String) -> String {
    if name.len() <= MAX_WIRE_TOOL_NAME_BYTES {
        return name;
    }
    let digest = fnv1a_hex(name.as_bytes());
    // The digest and its separator take the last bytes of the name.
    let keep = MAX_WIRE_TOOL_NAME_BYTES - digest.len() - 1;
    let mut head = String::with_capacity(MAX_WIRE_TOOL_NAME_BYTES);
    for character in name.chars() {
        if head.len() + character.len_utf8() > keep {
            break;
        }
        head.push(character);
    }
    format!("{head}_{digest}")
}

/// FNV-1a, written out because the shortened name of a tool has to stay the
/// same across builds, and the standard hasher does not promise that.
fn fnv1a_hex(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// How the flat names this crate sent map back to the tools the model called.
///
/// These protocols answer with a bare function name and no namespace, so the
/// decoder needs the mapping that was built while rendering the request to hand
/// the agent loop the tool identity it actually exposed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolAliases {
    by_wire_name: BTreeMap<String, ToolName>,
}

impl ToolAliases {
    /// The tool a flat wire name stands for, when this request exposed it.
    pub fn resolve(&self, wire_name: &str) -> Option<&ToolName> {
        self.by_wire_name.get(wire_name)
    }

    /// The tool a flat wire name stands for, falling back to a plain name when
    /// this request never aliased it.
    pub fn resolve_or_plain(&self, wire_name: &str) -> ToolName {
        self.resolve(wire_name)
            .cloned()
            .unwrap_or_else(|| ToolName::plain(wire_name))
    }

    fn insert(&mut self, wire_name: String, tool_name: ToolName) {
        self.by_wire_name.insert(wire_name, tool_name);
    }
}

/// One tool as these protocols carry it: a single function with a JSON schema.
#[derive(Debug, Clone, PartialEq)]
pub struct WireTool {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// Flatten the canonical tool list into the function tools these protocols
/// carry, recording how each flat name maps back.
///
/// Namespaced tools are the reason this exists: every MCP tool is a namespace
/// holding one function, and dropping the namespace would either hide the tool
/// or leave the model calling a name no registry entry answers to. Hosted web
/// search and freeform tools stay on the Responses path, because neither
/// protocol has a field for a tool the provider runs itself or one that takes
/// freeform input.
pub fn wire_tools(tools: &[ToolSpec]) -> Result<(Vec<WireTool>, ToolAliases), ProviderError> {
    let mut used: BTreeSet<String> = BTreeSet::new();
    let mut aliases = ToolAliases::default();
    let mut rendered = Vec::new();

    for tool in tools {
        match tool {
            ToolSpec::Function(function) => {
                // A plain function keeps its own name, so a decoded call needs
                // no alias to resolve it.
                rendered.push(WireTool {
                    name: unique_name(&mut used, function.name.clone()),
                    description: function.description.clone(),
                    parameters: schema(&function.parameters)?,
                });
            }
            ToolSpec::Namespace(namespace) => {
                for nested in &namespace.tools {
                    let ResponsesApiNamespaceTool::Function(function) = nested else {
                        // A freeform member has no function schema to send.
                        continue;
                    };
                    let wire_name = unique_name(
                        &mut used,
                        wire_tool_name(Some(namespace.name.as_str()), &function.name),
                    );
                    aliases.insert(
                        wire_name.clone(),
                        ToolName::namespaced(namespace.name.clone(), function.name.clone()),
                    );
                    rendered.push(WireTool {
                        name: wire_name,
                        // The namespace description is what tells the model which
                        // server a tool belongs to, and a flat tool has no field
                        // of its own for it, so it opens the description.
                        description: described_by(&namespace.description, &function.description),
                        parameters: schema(&function.parameters)?,
                    });
                }
            }
            ToolSpec::ToolSearch {
                description,
                parameters,
                ..
            } => {
                // Discovery is a client-run tool, and a function call is the
                // only way these protocols can ask for it.
                let wire_name = unique_name(&mut used, TOOL_SEARCH_TOOL_NAME.to_string());
                aliases.insert(wire_name.clone(), ToolName::plain(TOOL_SEARCH_TOOL_NAME));
                rendered.push(WireTool {
                    name: wire_name,
                    description: description.clone(),
                    parameters: schema(parameters)?,
                });
            }
            ToolSpec::WebSearch { .. } | ToolSpec::Freeform(_) => {}
        }
    }

    Ok((rendered, aliases))
}

/// Keeps a wire name unique, since two namespaces can flatten to one name.
fn unique_name(used: &mut BTreeSet<String>, candidate: String) -> String {
    let candidate = fit_wire_name(candidate);
    if used.insert(candidate.clone()) {
        return candidate;
    }
    let mut suffix = 2;
    loop {
        let name = fit_wire_name(format!("{candidate}_{suffix}"));
        if used.insert(name.clone()) {
            return name;
        }
        suffix += 1;
    }
}

/// A tool description that carries the namespace it came from.
fn described_by(namespace_description: &str, description: &str) -> String {
    let namespace_description = namespace_description.trim();
    if namespace_description.is_empty() {
        return description.to_string();
    }
    if description.trim().is_empty() {
        return namespace_description.to_string();
    }
    format!("{namespace_description}\n\n{description}")
}

/// Tools in the nested shape the Chat Completions API expects.
pub fn openai_function_tools(tools: &[WireTool]) -> Vec<ChatCompletionTools> {
    tools
        .iter()
        .map(|tool| {
            ChatCompletionTools::Function(ChatCompletionTool {
                function: FunctionObject {
                    name: tool.name.clone(),
                    description: Some(tool.description.clone()),
                    parameters: Some(tool.parameters.clone()),
                    strict: None,
                },
            })
        })
        .collect()
}

/// Tools in the shape the Anthropic Messages API expects.
pub fn anthropic_function_tools(tools: &[WireTool]) -> Vec<ToolParam> {
    tools
        .iter()
        .map(|tool| {
            let mut rendered = ToolParam::new(tool.name.clone(), tool.parameters.clone());
            rendered.description = Some(tool.description.clone());
            rendered
        })
        .collect()
}

fn schema(parameters: &JsonSchema) -> Result<Value, ProviderError> {
    serde_json::to_value(parameters).map_err(|source| ProviderError::Serialize {
        context: "tool schema",
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_tools::ResponsesApiNamespace;
    use codex_tools::ResponsesApiTool;

    fn function_tool(name: &str) -> ToolSpec {
        function_spec(name, format!("{name} tool"))
    }

    fn function_spec(name: &str, description: String) -> ToolSpec {
        ToolSpec::Function(ResponsesApiTool {
            name: name.to_string(),
            description,
            strict: false,
            defer_loading: None,
            parameters: JsonSchema::default(),
            output_schema: None,
        })
    }

    fn namespace_tool(namespace: &str, name: &str) -> ToolSpec {
        ToolSpec::Namespace(ResponsesApiNamespace {
            name: namespace.to_string(),
            description: format!("{namespace} tools"),
            tools: vec![ResponsesApiNamespaceTool::Function(ResponsesApiTool {
                name: name.to_string(),
                description: format!("{namespace}{name} tool"),
                strict: false,
                defer_loading: None,
                parameters: JsonSchema::default(),
                output_schema: None,
            })],
        })
    }

    #[test]
    fn completions_nests_the_function() {
        let (tools, _) = wire_tools(&[function_tool("shell")]).expect("flatten tools");
        let rendered = serde_json::to_value(&openai_function_tools(&tools)[0]).expect("serialize");
        assert_eq!(rendered["type"], "function");
        assert_eq!(rendered["function"]["name"], "shell");
        assert!(rendered["function"]["parameters"].is_object());
    }

    #[test]
    fn anthropic_flattens_the_definition() {
        let (tools, _) = wire_tools(&[function_tool("shell")]).expect("flatten tools");
        let rendered =
            serde_json::to_value(&anthropic_function_tools(&tools)[0]).expect("serialize");
        assert_eq!(rendered["name"], "shell");
        assert!(rendered["input_schema"].is_object());
        // The provider takes custom tools untagged.
        assert!(rendered.get("type").is_none());
        assert!(rendered.get("function").is_none());
    }

    #[test]
    fn a_namespaced_tool_crosses_over_under_one_flat_name() {
        let (tools, aliases) =
            wire_tools(&[namespace_tool("mcp__sample__", "search")]).expect("flatten tools");

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "mcp__sample__search");
        // The namespace description is the only thing that says which server a
        // tool belongs to, and a flat tool has nowhere else to put it.
        assert_eq!(
            tools[0].description,
            "mcp__sample__ tools\n\nmcp__sample__search tool"
        );
        assert_eq!(
            aliases.resolve("mcp__sample__search"),
            Some(&ToolName::namespaced("mcp__sample__", "search"))
        );
        assert_eq!(aliases.resolve("other"), None);
    }

    #[test]
    fn a_plain_namespace_is_joined_with_a_delimiter() {
        let (tools, aliases) = wire_tools(&[namespace_tool("web", "run")]).expect("flatten tools");

        assert_eq!(tools[0].name, "web__run");
        assert_eq!(
            aliases.resolve("web__run"),
            Some(&ToolName::namespaced("web", "run"))
        );
    }

    #[test]
    fn tool_search_crosses_over_as_a_function() {
        let (tools, aliases) = wire_tools(&[ToolSpec::ToolSearch {
            execution: "client".to_string(),
            description: "Searches deferred tools.".to_string(),
            parameters: JsonSchema::default(),
        }])
        .expect("flatten tools");

        assert_eq!(tools[0].name, "tool_search");
        assert_eq!(
            aliases.resolve("tool_search"),
            Some(&ToolName::plain("tool_search"))
        );
    }

    #[test]
    fn hosted_and_freeform_tools_stay_behind() {
        let freeform = ToolSpec::Freeform(codex_tools::FreeformTool {
            name: "apply_patch".to_string(),
            description: "patch".to_string(),
            defer_loading: None,
            format: codex_tools::FreeformToolFormat {
                r#type: "grammar".to_string(),
                syntax: "lark".to_string(),
                definition: "start: patch".to_string(),
            },
        });
        let (tools, aliases) = wire_tools(&[freeform]).expect("flatten tools");

        assert!(tools.is_empty());
        assert_eq!(aliases.resolve("apply_patch"), None);
    }

    #[test]
    fn an_overlong_name_is_shortened_deterministically() {
        let long_name = "create_entities_that_live_in_a_very_long_namespace".repeat(2);
        let shortened = wire_tool_name(Some("mcp__sample__"), &long_name);

        assert_eq!(shortened.len(), MAX_WIRE_TOOL_NAME_BYTES);
        assert!(shortened.starts_with("mcp__sample__create_entities"));
        // The same tool always shortens to the same name, and a name that shares
        // its head still ends up apart.
        assert_eq!(shortened, wire_tool_name(Some("mcp__sample__"), &long_name));
        let mut nearly_the_same = long_name.clone();
        nearly_the_same.push('x');
        assert_ne!(
            shortened,
            wire_tool_name(Some("mcp__sample__"), &nearly_the_same)
        );
        assert_eq!(wire_tool_name(None, "shell"), "shell");
    }

    #[test]
    fn a_flattened_name_that_collides_keeps_its_alias() {
        let (tools, aliases) =
            wire_tools(&[function_tool("web__run"), namespace_tool("web", "run")])
                .expect("flatten tools");

        assert_eq!(tools[0].name, "web__run");
        assert_eq!(tools[1].name, "web__run_2");
        assert_eq!(
            aliases.resolve("web__run_2"),
            Some(&ToolName::namespaced("web", "run"))
        );
    }
}
