# extra-ca

Optional. Place extra CA certificates (PEM, `*.crt`) here before running
`docker build` and the TensorFlow image will trust them (system store, Java
truststore for Bazel, pip). Needed behind a TLS-inspecting proxy, where Bazel
fails with `PKIX path building failed`.

On a host that already trusts the CA, e.g.:

    cp /usr/local/share/ca-certificates/<corp-root>.crt extra-ca/

Certificate files are git-ignored.
