"""Microsoft Teams provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, SecretStr, model_validator


class TeamsConfig(BaseModel):
    """Microsoft Teams Bot Framework configuration.

    Supports two authentication modes:

    **Client secret (password)**::

        TeamsConfig(app_id="...", app_password="...")

    **Certificate-based**::

        TeamsConfig(
            app_id="...",
            certificate_thumbprint="AB01CD...",
            certificate_private_key="-----BEGIN RSA PRIVATE KEY-----\\n...",
        )

    Exactly one mode must be provided.
    """

    app_id: str
    app_password: SecretStr | None = None
    tenant_id: str = "common"

    # Certificate-based auth fields
    certificate_thumbprint: str | None = None
    certificate_private_key: SecretStr | None = None
    certificate_public: str | None = None

    @model_validator(mode="after")
    def _validate_auth_mode(self) -> TeamsConfig:
        has_password = self.app_password is not None
        has_thumbprint = self.certificate_thumbprint is not None
        has_key = self.certificate_private_key is not None

        if has_password and (has_thumbprint or has_key):
            msg = (
                "Cannot specify both app_password and certificate fields. "
                "Use either password auth or certificate auth, not both."
            )
            raise ValueError(msg)

        if has_thumbprint != has_key:
            missing = "certificate_private_key" if has_thumbprint else "certificate_thumbprint"
            msg = (
                f"Certificate auth requires both certificate_thumbprint "
                f"and certificate_private_key (missing {missing})."
            )
            raise ValueError(msg)

        if not has_password and not has_thumbprint:
            msg = (
                "Either app_password (for password auth) or "
                "certificate_thumbprint + certificate_private_key "
                "(for certificate auth) must be provided."
            )
            raise ValueError(msg)

        if self.certificate_public is not None and not has_thumbprint:
            msg = "certificate_public is only valid with certificate auth."
            raise ValueError(msg)

        return self

    @property
    def uses_certificate_auth(self) -> bool:
        """Return True if configured for certificate-based authentication."""
        return self.certificate_thumbprint is not None
