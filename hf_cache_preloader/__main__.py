import logging

from .app import main

logger = logging.getLogger(__name__)

raise SystemExit(main())
