"""OpenCode plugin generation follows the official hook signatures."""

from onecomp.compaction.adapters.opencode import render_plugin, write_plugin


def test_generated_plugin_uses_occurrence_identity_and_mutable_output(tmp_path):
    source = render_plugin(
        python_executable="/opt/python",
        result_max_chars=321,
        max_archive_bytes=4096,
    )
    assert '"tool.execute.after"' in source
    assert '"experimental.session.compacting"' in source
    assert "input.sessionID" in source
    assert "input.callID" in source
    assert "output.output = replacement" in source
    assert 'stderr: "inherit"' in source
    assert "Fail open" in source
    assert "catch (error)" in source
    assert "resultMaxChars = 321" in source
    assert "maxArchiveBytes = 4096" in source
    assert "hookTimeoutMs = 30_000" in source
    assert 'typeof output.output !== "string"' in source
    assert "Array.isArray(output.context)" in source
    destination = write_plugin(tmp_path / "plugin" / "onecomp.ts")
    assert destination.is_file()
    assert not list(destination.parent.glob("*.tmp"))
