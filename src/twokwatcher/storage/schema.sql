-- 2kWatcher schema.
--
-- Shaped around the fact that this tracks a small, fixed roster of real people
-- across many games, rather than arbitrary NBA rosters. `players` is that
-- registry, and it is what makes squad analytics possible: once box score rows
-- are keyed to known gamertags, lineup +/- falls straight out of a GROUP BY.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- One run of the watcher, roughly "a night of playing".
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    source      TEXT,                -- 'virtualcam' | 'file'
    notes       TEXT
);

-- The friend registry. is_me flags your own gamertag.
CREATE TABLE IF NOT EXISTS players (
    id           INTEGER PRIMARY KEY,
    gamertag     TEXT NOT NULL UNIQUE,
    display_name TEXT,
    is_me        INTEGER NOT NULL DEFAULT 0,
    is_friend    INTEGER NOT NULL DEFAULT 0,
    first_seen   TEXT,
    last_seen    TEXT
);

CREATE TABLE IF NOT EXISTS games (
    id          INTEGER PRIMARY KEY,
    session_id  INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    mode        TEXT,                -- 'rec' | 'pro_am' | 'park' | 'mycareer' | ...
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    score_us    INTEGER,
    score_them  INTEGER,
    result      TEXT,                -- 'W' | 'L' | NULL if unknown
    video_path  TEXT                 -- recording this game came from, if any
);

-- One row per player per game: the parsed box score line.
CREATE TABLE IF NOT EXISTS game_players (
    id          INTEGER PRIMARY KEY,
    game_id     INTEGER NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    player_id   INTEGER NOT NULL REFERENCES players(id),
    team        TEXT NOT NULL,       -- 'us' | 'them'
    pts         INTEGER, reb INTEGER, ast INTEGER,
    stl         INTEGER, blk INTEGER, tov INTEGER,
    fgm         INTEGER, fga INTEGER,
    tpm         INTEGER, tpa INTEGER,
    ftm         INTEGER, fta INTEGER,
    grade       TEXT,                -- teammate grade, e.g. 'A+'
    plus_minus  INTEGER,
    UNIQUE (game_id, player_id)
);

-- Discrete things that happened, timestamped against the game clock.
-- kind: 'shot_attempt' | 'shot_feedback' | 'make' | 'miss' | 'block' | 'steal' | ...
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY,
    game_id     INTEGER REFERENCES games(id) ON DELETE CASCADE,
    player_id   INTEGER REFERENCES players(id),
    frame_index INTEGER,
    video_ts    REAL,                -- seconds into the source
    quarter     INTEGER,
    game_clock  TEXT,                -- as displayed, e.g. '4:32'
    kind        TEXT NOT NULL,
    payload     TEXT                 -- JSON: release timing, shot type, court xy
);

-- Every committed state transition. Cheap, and invaluable when a parser
-- misbehaves and you need to know what the watcher thought was on screen.
CREATE TABLE IF NOT EXISTS state_log (
    id          INTEGER PRIMARY KEY,
    session_id  INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    frame_index INTEGER,
    video_ts    REAL,
    previous    TEXT,
    current     TEXT
);

-- Reserved for the tracker. Positions are COURT coordinates in feet, origin at
-- centre court, not pixels — pixel positions are meaningless once the camera pans.
CREATE TABLE IF NOT EXISTS tracks (
    id          INTEGER PRIMARY KEY,
    game_id     INTEGER REFERENCES games(id) ON DELETE CASCADE,
    frame_index INTEGER NOT NULL,
    track_id    INTEGER NOT NULL,
    player_id   INTEGER REFERENCES players(id),
    team        TEXT,
    court_x     REAL,
    court_y     REAL,
    confidence  REAL
);

CREATE INDEX IF NOT EXISTS idx_events_game    ON events(game_id, kind);
CREATE INDEX IF NOT EXISTS idx_gp_game        ON game_players(game_id);
CREATE INDEX IF NOT EXISTS idx_gp_player      ON game_players(player_id);
CREATE INDEX IF NOT EXISTS idx_tracks_game    ON tracks(game_id, frame_index);
CREATE INDEX IF NOT EXISTS idx_state_session  ON state_log(session_id);
